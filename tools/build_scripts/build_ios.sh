#!/usr/bin/env bash
# =============================================================================
# Genovo Engine - iOS Cross-Compile Stub
# =============================================================================
# Prerequisites:
#   - macOS host with Xcode installed
#   - Rust targets: aarch64-apple-ios, aarch64-apple-ios-sim
#   - cargo-lipo or similar for fat binary generation
#
# TODO(BUILD): Implement iOS cross-compilation pipeline - Month 8-9
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_TYPE="${1:-release}"

echo "=== Genovo Engine - iOS Cross-Compile ==="
echo "  Project root: $PROJECT_ROOT"
echo "  Build type:   $BUILD_TYPE"
echo ""

# ---------------------------------------------------------------------------
# Verify host platform
# ---------------------------------------------------------------------------
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: iOS builds require macOS host."
    exit 1
fi

# ---------------------------------------------------------------------------
# Install Rust iOS targets
# ---------------------------------------------------------------------------
echo "--- Ensuring Rust iOS targets ---"
rustup target add aarch64-apple-ios 2>/dev/null || true
rustup target add aarch64-apple-ios-sim 2>/dev/null || true
echo ""

# ---------------------------------------------------------------------------
# Build for iOS device (ARM64)
# ---------------------------------------------------------------------------
echo "--- Building for iOS device (aarch64-apple-ios) ---"

cd "$PROJECT_ROOT"

# TODO(BUILD): Build only the crates that make sense for iOS runtime
# TODO(BUILD): Exclude editor, FFI desktop-specific code
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release --target aarch64-apple-ios || echo "iOS device build: NOT YET CONFIGURED"
else
    cargo build --target aarch64-apple-ios || echo "iOS device build: NOT YET CONFIGURED"
fi
echo ""

# ---------------------------------------------------------------------------
# Build for iOS Simulator (ARM64 sim)
# ---------------------------------------------------------------------------
echo "--- Building for iOS Simulator (aarch64-apple-ios-sim) ---"

if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release --target aarch64-apple-ios-sim || echo "iOS sim build: NOT YET CONFIGURED"
else
    cargo build --target aarch64-apple-ios-sim || echo "iOS sim build: NOT YET CONFIGURED"
fi
echo ""

# ---------------------------------------------------------------------------
# Cook assets for iOS
# ---------------------------------------------------------------------------
echo "--- Cooking assets for iOS ---"
# TODO(BUILD): Cook assets with ASTC textures, Metal shaders
# python3 "$PROJECT_ROOT/tools/asset_cooker/cook.py" \
#     --input "$PROJECT_ROOT/assets" \
#     --output "$PROJECT_ROOT/target/cooked-ios" \
#     --platform ios
echo "Asset cooking for iOS: TODO"
echo ""

# ---------------------------------------------------------------------------
# Generate Xcode project
# ---------------------------------------------------------------------------
# TODO(BUILD): Generate Xcode project with correct signing, entitlements
echo "Xcode project generation: TODO"
echo ""

echo "=== iOS cross-compile complete ==="
