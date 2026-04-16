#!/usr/bin/env bash
# =============================================================================
# Genovo Engine - PlayStation Build Stub
# =============================================================================
# Prerequisites:
#   - Sony PlayStation Partners SDK (under NDA)
#   - PlayStation development hardware (devkit/testkit)
#   - Registered PlayStation Partners developer account
#   - Platform-specific toolchain and sysroot
#
# TODO(BUILD): Implement PlayStation build pipeline - requires SDK under NDA
# TODO(BUILD): PlayStation SDK distribution is restricted - cannot include tooling
# =============================================================================

set -euo pipefail

echo "=== Genovo Engine - PlayStation Build ==="
echo ""
echo "ERROR: PlayStation builds require the Sony PlayStation Partners SDK."
echo ""
echo "Prerequisites:"
echo "  1. Register as a PlayStation Partners developer"
echo "     https://partners.playstation.net/"
echo "  2. Obtain and install the PlayStation SDK"
echo "  3. Set up the development environment per Sony documentation"
echo "  4. Configure environment variables:"
echo "       - SCE_ROOT_DIR"
echo "       - SCE_ORBIS_SDK_DIR (PS4) or SCE_PROSPERO_SDK_DIR (PS5)"
echo ""
echo "Once the SDK is available, this script will:"
echo "  1. Cross-compile Rust crates for PlayStation target"
echo "  2. Build C++ FFI layer with PlayStation SDK headers"
echo "  3. Cook assets for PlayStation (GNF textures, PSSL shaders)"
echo "  4. Package into PlayStation submission format"
echo "  5. Deploy to devkit via target manager"
echo ""
echo "NOTE: All PlayStation-specific code and build tooling falls under"
echo "      Sony's NDA. Do not commit SDK paths or platform-specific"
echo "      implementation details to public repositories."
echo ""
echo "Contact the engine team for PlayStation integration support."
echo ""

# TODO(BUILD): Implement when PlayStation SDK is available:
#
# PS_SDK="${SCE_PROSPERO_SDK_DIR:-}"
# if [ -z "$PS_SDK" ]; then
#     echo "SCE_PROSPERO_SDK_DIR not set"; exit 1
# fi
#
# # Cross-compile with PlayStation toolchain
# # (requires custom Rust target spec JSON)
# cargo build --release --target=x86_64-scei-ps5
#
# # Build C++ with PlayStation CMake toolchain
# cmake -S crates/genovo-ffi -B target/cmake-ps5 \
#     -DCMAKE_TOOLCHAIN_FILE="$PS_SDK/host_tools/lib/cmake/PS5Toolchain.cmake"
#
# # Package for submission
# orbis-pub-cmd img_create --odir target/ps5-package

exit 1
