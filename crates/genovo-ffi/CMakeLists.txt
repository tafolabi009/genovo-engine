cmake_minimum_required(VERSION 3.20)
project(genovo-ffi-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Options
# =============================================================================

option(GENOVO_USE_PHYSX "Enable NVIDIA PhysX integration" OFF)
option(GENOVO_USE_FMOD "Enable FMOD Studio audio integration" OFF)
option(GENOVO_USE_WWISE "Enable Audiokinetic Wwise audio integration" OFF)

# =============================================================================
# Platform detection
# =============================================================================

if(WIN32)
    set(GENOVO_PLATFORM "windows")
    set(GENOVO_LIB_SUFFIX ".lib")
    set(GENOVO_DLL_SUFFIX ".dll")
elseif(APPLE)
    set(GENOVO_PLATFORM "macos")
    set(GENOVO_LIB_SUFFIX ".a")
    set(GENOVO_DLL_SUFFIX ".dylib")
elseif(UNIX)
    set(GENOVO_PLATFORM "linux")
    set(GENOVO_LIB_SUFFIX ".a")
    set(GENOVO_DLL_SUFFIX ".so")
else()
    message(WARNING "Unsupported platform")
endif()

message(STATUS "Genovo FFI building for platform: ${GENOVO_PLATFORM}")

# =============================================================================
# Rust FFI library (built by Cargo, linked here)
# =============================================================================

# The Rust crate produces libgenovo_ffi.a / genovo_ffi.lib
# Locate it in the Cargo target directory.
set(CARGO_TARGET_DIR "${CMAKE_SOURCE_DIR}/../../target" CACHE PATH "Cargo target directory")
set(CARGO_PROFILE "release" CACHE STRING "Cargo build profile (debug/release)")

if(WIN32)
    set(RUST_FFI_LIB "${CARGO_TARGET_DIR}/${CARGO_PROFILE}/genovo_ffi.lib")
else()
    set(RUST_FFI_LIB "${CARGO_TARGET_DIR}/${CARGO_PROFILE}/libgenovo_ffi.a")
endif()

# =============================================================================
# PhysX integration
# =============================================================================

if(GENOVO_USE_PHYSX)
    # TODO(FFI): Set PHYSX_ROOT to actual PhysX SDK install path
    set(PHYSX_ROOT "" CACHE PATH "Path to PhysX SDK root")

    if(NOT PHYSX_ROOT)
        message(FATAL_ERROR "PHYSX_ROOT must be set when GENOVO_USE_PHYSX is ON")
    endif()

    set(PHYSX_INCLUDE_DIR "${PHYSX_ROOT}/include")
    set(PHYSX_LIB_DIR "${PHYSX_ROOT}/bin/${GENOVO_PLATFORM}")

    find_library(PHYSX_LIB NAMES PhysX_64 PhysX PATHS "${PHYSX_LIB_DIR}" NO_DEFAULT_PATH)
    find_library(PHYSX_COMMON_LIB NAMES PhysXCommon_64 PhysXCommon PATHS "${PHYSX_LIB_DIR}" NO_DEFAULT_PATH)
    find_library(PHYSX_FOUNDATION_LIB NAMES PhysXFoundation_64 PhysXFoundation PATHS "${PHYSX_LIB_DIR}" NO_DEFAULT_PATH)
    find_library(PHYSX_COOKING_LIB NAMES PhysXCooking_64 PhysXCooking PATHS "${PHYSX_LIB_DIR}" NO_DEFAULT_PATH)
    find_library(PHYSX_EXTENSIONS_LIB NAMES PhysXExtensions_64 PhysXExtensions PATHS "${PHYSX_LIB_DIR}" NO_DEFAULT_PATH)

    add_library(physx_bridge STATIC
        # TODO(FFI): Add C++ PhysX bridge source files here
        # src/cpp/physx_bridge.cpp
    )
    target_include_directories(physx_bridge PUBLIC "${PHYSX_INCLUDE_DIR}")
    target_link_libraries(physx_bridge PUBLIC
        ${PHYSX_LIB}
        ${PHYSX_COMMON_LIB}
        ${PHYSX_FOUNDATION_LIB}
        ${PHYSX_COOKING_LIB}
        ${PHYSX_EXTENSIONS_LIB}
    )

    message(STATUS "PhysX integration enabled (root: ${PHYSX_ROOT})")
endif()

# =============================================================================
# FMOD Studio integration
# =============================================================================

if(GENOVO_USE_FMOD)
    # TODO(FFI): Set FMOD_ROOT to actual FMOD Studio SDK install path
    set(FMOD_ROOT "" CACHE PATH "Path to FMOD Studio SDK root")

    if(NOT FMOD_ROOT)
        message(FATAL_ERROR "FMOD_ROOT must be set when GENOVO_USE_FMOD is ON")
    endif()

    set(FMOD_INCLUDE_DIR "${FMOD_ROOT}/api/studio/inc" "${FMOD_ROOT}/api/core/inc")
    set(FMOD_LIB_DIR "${FMOD_ROOT}/api/studio/lib/${GENOVO_PLATFORM}" "${FMOD_ROOT}/api/core/lib/${GENOVO_PLATFORM}")

    find_library(FMOD_CORE_LIB NAMES fmod fmod_vc PATHS ${FMOD_LIB_DIR} NO_DEFAULT_PATH)
    find_library(FMOD_STUDIO_LIB NAMES fmodstudio fmodstudio_vc PATHS ${FMOD_LIB_DIR} NO_DEFAULT_PATH)

    add_library(fmod_bridge STATIC
        # TODO(FFI): Add C++ FMOD bridge source files here
        # src/cpp/fmod_bridge.cpp
    )
    target_include_directories(fmod_bridge PUBLIC ${FMOD_INCLUDE_DIR})
    target_link_libraries(fmod_bridge PUBLIC ${FMOD_CORE_LIB} ${FMOD_STUDIO_LIB})

    message(STATUS "FMOD Studio integration enabled (root: ${FMOD_ROOT})")
endif()

# =============================================================================
# Wwise integration
# =============================================================================

if(GENOVO_USE_WWISE)
    # TODO(FFI): Set WWISE_ROOT to actual Wwise SDK install path
    set(WWISE_ROOT "" CACHE PATH "Path to Wwise SDK root")

    if(NOT WWISE_ROOT)
        message(FATAL_ERROR "WWISE_ROOT must be set when GENOVO_USE_WWISE is ON")
    endif()

    message(STATUS "Wwise integration enabled (root: ${WWISE_ROOT})")
    # TODO(FFI): Configure Wwise include/lib paths - Month 4-5
endif()

# =============================================================================
# Combined FFI library target
# =============================================================================

# This target links the Rust FFI static library with any C++ bridge libraries
# into a single output that game projects can consume.

# TODO(FFI): Uncomment and configure when bridge sources exist
# add_library(genovo_ffi_combined STATIC)
# target_link_libraries(genovo_ffi_combined PUBLIC ${RUST_FFI_LIB})
# if(GENOVO_USE_PHYSX)
#     target_link_libraries(genovo_ffi_combined PUBLIC physx_bridge)
# endif()
# if(GENOVO_USE_FMOD)
#     target_link_libraries(genovo_ffi_combined PUBLIC fmod_bridge)
# endif()

# =============================================================================
# Install rules
# =============================================================================

# install(TARGETS genovo_ffi_combined ARCHIVE DESTINATION lib)
# install(FILES "${CMAKE_SOURCE_DIR}/include/genovo_ffi.h" DESTINATION include)
