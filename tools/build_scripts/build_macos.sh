#!/usr/bin/env bash
# =============================================================================
# Genovo Engine - macOS Build Script
# =============================================================================
# Prerequisites:
#   - Rust toolchain (rustup, cargo) with stable-aarch64-apple-darwin or
#     stable-x86_64-apple-darwin
#   - CMake 3.20+
#   - Xcode Command Line Tools
#   - Vulkan SDK (MoltenVK) for Vulkan on macOS
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_TYPE="${1:-release}"

echo "=== Genovo Engine - macOS Build ==="
echo "  Project root: $PROJECT_ROOT"
echo "  Build type:   $BUILD_TYPE"
echo "  Architecture: $(uname -m)"
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
# Step 2: Build C++ FFI layer
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
          -G "Unix Makefiles" \
          -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
          -DCARGO_TARGET_DIR="$PROJECT_ROOT/target" \
          -DCARGO_PROFILE="$BUILD_TYPE"

    cmake --build "$FFI_BUILD_DIR" --parallel "$(sysctl -n hw.ncpu)"

    echo "C++ FFI build complete."
else
    echo "Skipping C++ FFI build (no CMakeLists.txt found)."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Compile shaders (Metal via SPIR-V cross-compilation)
# ---------------------------------------------------------------------------
if command -v glslangValidator &> /dev/null; then
    echo "--- Compiling shaders ---"
    # TODO(BUILD): Compile shaders to Metal Shading Language via SPIR-V
    # python3 "$PROJECT_ROOT/tools/shader_compiler/compile.py" \
    #     --input "$PROJECT_ROOT/assets/shaders" \
    #     --output "$PROJECT_ROOT/target/shaders" \
    #     --target metal
    echo "Shader compilation: TODO"
else
    echo "Skipping shader compilation (glslangValidator not found)."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Cook assets
# ---------------------------------------------------------------------------
# TODO(BUILD): Integrate asset cooking for macOS
echo "Asset cooking: TODO"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Run tests
# ---------------------------------------------------------------------------
echo "--- Running tests ---"
cargo test --workspace
echo ""

echo "=== macOS build complete ==="
