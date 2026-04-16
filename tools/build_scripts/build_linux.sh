#!/usr/bin/env bash
# =============================================================================
# Genovo Engine - Linux Build Script
# =============================================================================
# Prerequisites:
#   - Rust toolchain (rustup, cargo)
#   - CMake 3.20+
#   - GCC 12+ or Clang 15+
#   - Vulkan SDK (libvulkan-dev, vulkan-tools)
#   - X11/Wayland development headers
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_TYPE="${1:-release}"
NUM_CORES="$(nproc)"

echo "=== Genovo Engine - Linux Build ==="
echo "  Project root: $PROJECT_ROOT"
echo "  Build type:   $BUILD_TYPE"
echo "  CPU cores:    $NUM_CORES"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Check prerequisites
# ---------------------------------------------------------------------------
echo "--- Checking prerequisites ---"
command -v cargo >/dev/null 2>&1 || { echo "ERROR: cargo not found. Install Rust: https://rustup.rs"; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found. Install cmake."; exit 1; }
echo "Prerequisites OK."
echo ""

# ---------------------------------------------------------------------------
# Step 2: Build Rust crates
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
# Step 3: Build C++ FFI layer
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

    cmake --build "$FFI_BUILD_DIR" --parallel "$NUM_CORES"

    echo "C++ FFI build complete."
else
    echo "Skipping C++ FFI build (no CMakeLists.txt found)."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Compile shaders
# ---------------------------------------------------------------------------
if command -v glslangValidator &> /dev/null; then
    echo "--- Compiling shaders ---"
    # TODO(BUILD): Integrate shader compilation
    # python3 "$PROJECT_ROOT/tools/shader_compiler/compile.py" \
    #     --input "$PROJECT_ROOT/assets/shaders" \
    #     --output "$PROJECT_ROOT/target/shaders" \
    #     --target vulkan
    echo "Shader compilation: TODO"
else
    echo "Skipping shader compilation (glslangValidator not found)."
    echo "Install Vulkan SDK: https://vulkan.lunarg.com/sdk/home"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 5: Cook assets
# ---------------------------------------------------------------------------
# TODO(BUILD): Integrate asset cooking for Linux
echo "Asset cooking: TODO"
echo ""

# ---------------------------------------------------------------------------
# Step 6: Run tests
# ---------------------------------------------------------------------------
echo "--- Running tests ---"
cargo test --workspace
echo ""

echo "=== Linux build complete ==="
