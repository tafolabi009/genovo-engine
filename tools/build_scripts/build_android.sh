#!/usr/bin/env bash
# =============================================================================
# Genovo Engine - Android NDK Build Stub
# =============================================================================
# Prerequisites:
#   - Android NDK (r26+) with ANDROID_NDK_HOME set
#   - Rust targets: aarch64-linux-android, armv7-linux-androideabi
#   - cargo-ndk for simplified NDK builds
#
# TODO(BUILD): Implement Android NDK cross-compilation pipeline - Month 9-10
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_TYPE="${1:-release}"
MIN_SDK_VERSION="${ANDROID_MIN_SDK:-24}"

echo "=== Genovo Engine - Android NDK Build ==="
echo "  Project root:      $PROJECT_ROOT"
echo "  Build type:        $BUILD_TYPE"
echo "  Min SDK version:   $MIN_SDK_VERSION"
echo "  NDK home:          ${ANDROID_NDK_HOME:-NOT SET}"
echo ""

# ---------------------------------------------------------------------------
# Verify NDK
# ---------------------------------------------------------------------------
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    echo "ERROR: ANDROID_NDK_HOME environment variable is not set."
    echo "  Download NDK: https://developer.android.com/ndk/downloads"
    exit 1
fi

if [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "ERROR: ANDROID_NDK_HOME does not exist: $ANDROID_NDK_HOME"
    exit 1
fi

# ---------------------------------------------------------------------------
# Install Rust Android targets
# ---------------------------------------------------------------------------
echo "--- Ensuring Rust Android targets ---"
rustup target add aarch64-linux-android 2>/dev/null || true
rustup target add armv7-linux-androideabi 2>/dev/null || true
rustup target add x86_64-linux-android 2>/dev/null || true
echo ""

# ---------------------------------------------------------------------------
# Check for cargo-ndk
# ---------------------------------------------------------------------------
if ! command -v cargo-ndk &> /dev/null; then
    echo "WARNING: cargo-ndk not found. Install with: cargo install cargo-ndk"
    echo "  Falling back to manual NDK configuration."
fi

# ---------------------------------------------------------------------------
# Build for ARM64 (primary target)
# ---------------------------------------------------------------------------
echo "--- Building for Android ARM64 (aarch64-linux-android) ---"

cd "$PROJECT_ROOT"

# TODO(BUILD): Configure linker, sysroot, and target-specific flags
# TODO(BUILD): Build only runtime crates (no editor, no desktop platform)
if command -v cargo-ndk &> /dev/null; then
    if [ "$BUILD_TYPE" = "release" ]; then
        cargo ndk --target aarch64-linux-android --platform "$MIN_SDK_VERSION" build --release \
            || echo "Android ARM64 build: NOT YET CONFIGURED"
    else
        cargo ndk --target aarch64-linux-android --platform "$MIN_SDK_VERSION" build \
            || echo "Android ARM64 build: NOT YET CONFIGURED"
    fi
else
    echo "Android ARM64 build: REQUIRES cargo-ndk"
fi
echo ""

# ---------------------------------------------------------------------------
# Build for ARMv7 (legacy support, optional)
# ---------------------------------------------------------------------------
echo "--- Building for Android ARMv7 (optional) ---"
# TODO(BUILD): ARMv7 build for older devices
echo "ARMv7 build: TODO (optional legacy support)"
echo ""

# ---------------------------------------------------------------------------
# Cook assets for Android
# ---------------------------------------------------------------------------
echo "--- Cooking assets for Android ---"
# TODO(BUILD): Cook assets with ASTC/ETC2 textures, SPIR-V shaders
# python3 "$PROJECT_ROOT/tools/asset_cooker/cook.py" \
#     --input "$PROJECT_ROOT/assets" \
#     --output "$PROJECT_ROOT/target/cooked-android" \
#     --platform android
echo "Asset cooking for Android: TODO"
echo ""

# ---------------------------------------------------------------------------
# Package into AAR / APK
# ---------------------------------------------------------------------------
# TODO(BUILD): Generate Android library (AAR) or example APK via Gradle
echo "Android packaging: TODO"
echo ""

echo "=== Android NDK build complete ==="
