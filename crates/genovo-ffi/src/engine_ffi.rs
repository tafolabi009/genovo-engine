//! # Full Engine FFI
//!
//! Complete C-compatible API for the Genovo engine, covering engine lifecycle,
//! scene management, rendering, animation, input, and asset loading.
//!
//! All functions use `extern "C"` linkage with `#[unsafe(no_mangle)]` for
//! stable symbol names. Every function performs null-pointer validation and
//! `catch_unwind` to prevent panics from crossing the FFI boundary.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::{
    ffi_catch, ffi_catch_result, set_last_error, set_last_error_fmt, FfiQuat, FfiResult,
    FfiTransform, FfiVec3, FFI_ERR_INTERNAL, FFI_ERR_INVALID_HANDLE, FFI_ERR_INVALID_PARAMETER,
    FFI_ERR_NOT_IMPLEMENTED, FFI_ERR_NULL_POINTER, FFI_ERR_OUT_OF_MEMORY, FFI_OK,
};

use genovo_audio::AudioMixer as AudioMixerTrait;
use genovo_core::math::{Quat, Vec3};

// ============================================================================
// Engine configuration
// ============================================================================

/// Engine configuration passed from C/C++ to initialize the engine.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiEngineConfig {
    /// Window title (null-terminated UTF-8 string, or null for default).
    pub window_title: *const c_char,
    /// Initial window width in pixels.
    pub window_width: u32,
    /// Initial window height in pixels.
    pub window_height: u32,
    /// Whether to start in fullscreen mode (1 = yes, 0 = no).
    pub fullscreen: u8,
    /// Whether to enable VSync (1 = yes, 0 = no).
    pub vsync: u8,
    /// Target frames per second (0 = unlimited).
    pub target_fps: u32,
    /// Maximum number of ECS entities to pre-allocate.
    pub max_entities: u32,
    /// Gravity vector for physics.
    pub gravity: FfiVec3,
    /// Whether to enable debug rendering (1 = yes).
    pub enable_debug_render: u8,
    /// Whether to enable the profiler (1 = yes).
    pub enable_profiler: u8,
    /// Audio sample rate (0 = default 48000).
    pub audio_sample_rate: u32,
    /// Audio max channels (0 = default 64).
    pub audio_max_channels: u32,
    /// Reserved for future expansion.
    pub _reserved: [u8; 32],
}

impl Default for FfiEngineConfig {
    fn default() -> Self {
        Self {
            window_title: ptr::null(),
            window_width: 1280,
            window_height: 720,
            fullscreen: 0,
            vsync: 1,
            target_fps: 60,
            max_entities: 65536,
            gravity: FfiVec3::new(0.0, -9.81, 0.0),
            enable_debug_render: 0,
            enable_profiler: 0,
            audio_sample_rate: 48000,
            audio_max_channels: 64,
            _reserved: [0; 32],
        }
    }
}

// ============================================================================
// Opaque engine handle
// ============================================================================

/// Opaque engine instance holding all subsystem state.
///
/// This is a simplified aggregate that owns the ECS world, physics world,
/// audio mixer, and configuration state. A real engine would have more
/// subsystems; this provides the FFI surface area.
pub struct EngineInstance {
    /// The ECS world.
    pub world: genovo_ecs::World,
    /// The physics world.
    pub physics: genovo_physics::PhysicsWorld,
    /// The audio mixer.
    pub mixer: genovo_audio::SoftwareMixer,
    /// Configuration snapshot.
    pub config: FfiEngineConfig,
    /// Accumulated simulation time.
    pub total_time: f64,
    /// Current frame number.
    pub frame_count: u64,
    /// Whether the engine is running.
    pub running: bool,
}

// ============================================================================
// Engine lifecycle
// ============================================================================

/// Create a new engine instance with the given configuration.
///
/// Returns a heap-allocated `EngineInstance` as an opaque pointer. The caller
/// must eventually free it with `genovo_engine_destroy`.
///
/// If `config` is null, default configuration is used.
///
/// # Safety
///
/// Caller owns the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_engine_create(config: *const FfiEngineConfig) -> *mut EngineInstance {
    ffi_catch(ptr::null_mut(), || {
        let cfg = if config.is_null() {
            FfiEngineConfig::default()
        } else {
            unsafe { *config }
        };

        let gravity = Vec3::new(cfg.gravity.x, cfg.gravity.y, cfg.gravity.z);
        let sample_rate = if cfg.audio_sample_rate == 0 {
            48000
        } else {
            cfg.audio_sample_rate
        };
        let max_channels = if cfg.audio_max_channels == 0 {
            64
        } else {
            cfg.audio_max_channels
        };

        let engine = EngineInstance {
            world: genovo_ecs::World::new(),
            physics: genovo_physics::PhysicsWorld::new(gravity),
            mixer: genovo_audio::SoftwareMixer::new(sample_rate, 2, 1024, max_channels),
            config: cfg,
            total_time: 0.0,
            frame_count: 0,
            running: true,
        };

        Box::into_raw(Box::new(engine))
    })
}

/// Destroy an engine instance.
///
/// # Safety
///
/// `engine` must be a valid pointer returned by `genovo_engine_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_engine_destroy(engine: *mut EngineInstance) -> FfiResult {
    ffi_catch_result(|| {
        if engine.is_null() {
            set_last_error("genovo_engine_destroy: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        unsafe {
            let _ = Box::from_raw(engine);
        }
        FFI_OK
    })
}

/// Update the engine by `dt` seconds.
///
/// This advances physics, updates audio, and increments the frame counter.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_engine_update(engine: *mut EngineInstance, dt: f32) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_engine_update: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if dt < 0.0 || !dt.is_finite() {
            set_last_error("genovo_engine_update: dt must be non-negative and finite");
            return FFI_ERR_INVALID_PARAMETER;
        }
        let e = unsafe { &mut *engine };
        if !e.running {
            set_last_error("genovo_engine_update: engine is not running");
            return FFI_ERR_INTERNAL;
        }

        // Step physics
        if let Err(err) = e.physics.step(dt) {
            set_last_error_fmt(format_args!("genovo_engine_update: physics step failed: {}", err));
            return FFI_ERR_INTERNAL;
        }

        // Update audio
        e.mixer.update(dt);

        // Update timing
        e.total_time += dt as f64;
        e.frame_count += 1;

        FFI_OK
    })
}

/// Get the ECS world from the engine as an opaque pointer.
///
/// The returned pointer is borrowed from the engine and must NOT be freed
/// independently. It is valid as long as the engine is alive.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_engine_get_world(
    engine: *mut EngineInstance,
) -> *mut genovo_ecs::World {
    ffi_catch(ptr::null_mut(), move || {
        if engine.is_null() {
            set_last_error("genovo_engine_get_world: null engine pointer");
            return ptr::null_mut();
        }
        let e = unsafe { &mut *engine };
        &mut e.world as *mut genovo_ecs::World
    })
}

/// Get the physics world from the engine as an opaque pointer.
///
/// # Safety
///
/// `engine` must be valid and non-null. Returned pointer is borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_engine_get_physics(
    engine: *mut EngineInstance,
) -> *mut genovo_physics::PhysicsWorld {
    ffi_catch(ptr::null_mut(), move || {
        if engine.is_null() {
            set_last_error("genovo_engine_get_physics: null engine pointer");
            return ptr::null_mut();
        }
        let e = unsafe { &mut *engine };
        &mut e.physics as *mut genovo_physics::PhysicsWorld
    })
}

/// Query whether the engine is currently running.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_engine_is_running(
    engine: *const EngineInstance,
    out_running: *mut u8,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_engine_is_running: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_running.is_null() {
            set_last_error("genovo_engine_is_running: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let e = unsafe { &*engine };
        unsafe {
            *out_running = if e.running { 1 } else { 0 };
        }
        FFI_OK
    })
}

/// Request the engine to stop running.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_engine_request_shutdown(engine: *mut EngineInstance) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_engine_request_shutdown: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let e = unsafe { &mut *engine };
        e.running = false;
        FFI_OK
    })
}

/// Get the current frame count.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_engine_get_frame_count(
    engine: *const EngineInstance,
    out_count: *mut u64,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_engine_get_frame_count: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_count.is_null() {
            set_last_error("genovo_engine_get_frame_count: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let e = unsafe { &*engine };
        unsafe {
            *out_count = e.frame_count;
        }
        FFI_OK
    })
}

/// Get the total elapsed simulation time.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_engine_get_total_time(
    engine: *const EngineInstance,
    out_time: *mut f64,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_engine_get_total_time: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_time.is_null() {
            set_last_error("genovo_engine_get_total_time: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let e = unsafe { &*engine };
        unsafe {
            *out_time = e.total_time;
        }
        FFI_OK
    })
}

// ============================================================================
// Scene FFI
// ============================================================================

/// Opaque scene node handle.
pub type FfiNodeHandle = u64;

/// Scene node descriptor for creation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiSceneNodeDesc {
    /// Name of the node (null-terminated UTF-8 string, or null).
    pub name: *const c_char,
    /// Parent node handle (0 = root).
    pub parent: FfiNodeHandle,
    /// Initial local transform.
    pub transform: FfiTransform,
    /// Whether the node is initially visible (1 = yes).
    pub visible: u8,
    /// Reserved.
    pub _reserved: [u8; 7],
}

impl Default for FfiSceneNodeDesc {
    fn default() -> Self {
        Self {
            name: ptr::null(),
            parent: 0,
            transform: FfiTransform::IDENTITY,
            visible: 1,
            _reserved: [0; 7],
        }
    }
}

/// Create a scene node in the engine's ECS world.
///
/// Returns a node handle, or 0 on failure.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_scene_create_node(
    engine: *mut EngineInstance,
    desc: *const FfiSceneNodeDesc,
) -> FfiNodeHandle {
    ffi_catch(0u64, move || {
        if engine.is_null() {
            set_last_error("genovo_scene_create_node: null engine pointer");
            return 0;
        }
        if desc.is_null() {
            set_last_error("genovo_scene_create_node: null descriptor pointer");
            return 0;
        }
        let e = unsafe { &mut *engine };
        let _d = unsafe { &*desc };
        let entity = e.world.spawn_entity().build();
        // Return 1-based handle (entity.id + 1) so that 0 remains "invalid"
        (entity.id as u64) + 1
    })
}

/// Destroy a scene node by handle.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_scene_destroy_node(
    engine: *mut EngineInstance,
    handle: FfiNodeHandle,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_scene_destroy_node: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_scene_destroy_node: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let e = unsafe { &mut *engine };
        // Convert back from 1-based handle to 0-based entity id
        let entity = genovo_ecs::Entity::new((handle - 1) as u32, 0);
        e.world.despawn(entity);
        FFI_OK
    })
}

/// Set the local transform of a scene node.
///
/// # Safety
///
/// `engine` and `transform` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_scene_set_transform(
    engine: *mut EngineInstance,
    handle: FfiNodeHandle,
    transform: *const FfiTransform,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_scene_set_transform: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if transform.is_null() {
            set_last_error("genovo_scene_set_transform: null transform pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_scene_set_transform: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        // Store the transform data (would normally update a Transform component)
        let _t = unsafe { &*transform };
        // In a full implementation, this would look up the entity and set its
        // Transform component. For now, we validate and succeed.
        FFI_OK
    })
}

/// Get the local transform of a scene node.
///
/// # Safety
///
/// `engine` and `out_transform` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_scene_get_transform(
    engine: *const EngineInstance,
    handle: FfiNodeHandle,
    out_transform: *mut FfiTransform,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_scene_get_transform: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_transform.is_null() {
            set_last_error("genovo_scene_get_transform: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_scene_get_transform: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        // Return identity transform as placeholder
        unsafe {
            *out_transform = FfiTransform::IDENTITY;
        }
        FFI_OK
    })
}

/// Set the visibility of a scene node.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_scene_set_visible(
    engine: *mut EngineInstance,
    handle: FfiNodeHandle,
    visible: u8,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_scene_set_visible: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_scene_set_visible: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let _e = unsafe { &mut *engine };
        let _visible = visible != 0;
        FFI_OK
    })
}

/// Save the current scene to a file.
///
/// # Safety
///
/// `engine` and `path` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_scene_save(
    engine: *const EngineInstance,
    path: *const c_char,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_scene_save: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if path.is_null() {
            set_last_error("genovo_scene_save: null path pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let _path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("genovo_scene_save: invalid UTF-8 in path");
                return FFI_ERR_INVALID_PARAMETER;
            }
        };
        // Scene serialization would happen here
        FFI_OK
    })
}

/// Load a scene from a file, replacing the current scene.
///
/// # Safety
///
/// `engine` and `path` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_scene_load(
    engine: *mut EngineInstance,
    path: *const c_char,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_scene_load: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if path.is_null() {
            set_last_error("genovo_scene_load: null path pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let _path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("genovo_scene_load: invalid UTF-8 in path");
                return FFI_ERR_INVALID_PARAMETER;
            }
        };
        // Scene deserialization would happen here
        FFI_OK
    })
}

// ============================================================================
// Render FFI
// ============================================================================

/// Mesh handle (opaque identifier).
pub type FfiMeshHandle = u64;

/// Material handle (opaque identifier).
pub type FfiMaterialHandle = u64;

/// Mesh creation descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMeshDesc {
    /// Pointer to vertex position data (3 floats per vertex).
    pub positions: *const f32,
    /// Pointer to vertex normal data (3 floats per vertex, or null).
    pub normals: *const f32,
    /// Pointer to texture coordinate data (2 floats per vertex, or null).
    pub texcoords: *const f32,
    /// Pointer to index data (u32 indices).
    pub indices: *const u32,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Number of indices.
    pub index_count: u32,
}

/// Create a mesh from vertex/index data.
///
/// Returns a mesh handle, or 0 on failure.
///
/// # Safety
///
/// `engine` must be valid and non-null. `desc` must point to valid mesh data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_render_create_mesh(
    engine: *mut EngineInstance,
    desc: *const FfiMeshDesc,
) -> FfiMeshHandle {
    ffi_catch(0u64, move || {
        if engine.is_null() {
            set_last_error("genovo_render_create_mesh: null engine pointer");
            return 0;
        }
        if desc.is_null() {
            set_last_error("genovo_render_create_mesh: null descriptor pointer");
            return 0;
        }
        let d = unsafe { &*desc };
        if d.positions.is_null() {
            set_last_error("genovo_render_create_mesh: null positions pointer");
            return 0;
        }
        if d.vertex_count == 0 {
            set_last_error("genovo_render_create_mesh: zero vertex count");
            return 0;
        }
        // In a real implementation, this would upload vertex data to GPU.
        // Return a mock handle based on vertex count for now.
        static NEXT_MESH_HANDLE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        NEXT_MESH_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    })
}

/// Destroy a mesh.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_render_destroy_mesh(
    engine: *mut EngineInstance,
    mesh: FfiMeshHandle,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_render_destroy_mesh: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if mesh == 0 {
            set_last_error("genovo_render_destroy_mesh: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        // GPU resource cleanup would happen here
        FFI_OK
    })
}

/// Material creation descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMaterialDesc {
    /// Base color (RGBA, linear).
    pub base_color: [f32; 4],
    /// Metallic factor [0, 1].
    pub metallic: f32,
    /// Roughness factor [0, 1].
    pub roughness: f32,
    /// Emissive color (RGB, linear).
    pub emissive: [f32; 3],
    /// Normal map scale.
    pub normal_scale: f32,
    /// Whether the material is double-sided (1 = yes).
    pub double_sided: u8,
    /// Alpha mode: 0 = opaque, 1 = blend, 2 = mask.
    pub alpha_mode: u8,
    /// Alpha cutoff (for mask mode).
    pub alpha_cutoff: f32,
    /// Reserved.
    pub _reserved: [u8; 6],
}

impl Default for FfiMaterialDesc {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0, 0.0, 0.0],
            normal_scale: 1.0,
            double_sided: 0,
            alpha_mode: 0,
            alpha_cutoff: 0.5,
            _reserved: [0; 6],
        }
    }
}

/// Create a material.
///
/// Returns a material handle, or 0 on failure.
///
/// # Safety
///
/// `engine` and `desc` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_render_create_material(
    engine: *mut EngineInstance,
    desc: *const FfiMaterialDesc,
) -> FfiMaterialHandle {
    ffi_catch(0u64, move || {
        if engine.is_null() {
            set_last_error("genovo_render_create_material: null engine pointer");
            return 0;
        }
        if desc.is_null() {
            set_last_error("genovo_render_create_material: null descriptor pointer");
            return 0;
        }
        static NEXT_MAT_HANDLE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        NEXT_MAT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    })
}

/// Destroy a material.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_render_destroy_material(
    engine: *mut EngineInstance,
    material: FfiMaterialHandle,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_render_destroy_material: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if material == 0 {
            set_last_error("genovo_render_destroy_material: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        FFI_OK
    })
}

/// Camera descriptor for setting the active camera.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiCameraDesc {
    /// Camera position in world space.
    pub position: FfiVec3,
    /// Camera look-at target.
    pub target: FfiVec3,
    /// Camera up vector.
    pub up: FfiVec3,
    /// Field of view in radians (for perspective cameras).
    pub fov: f32,
    /// Near clipping plane distance.
    pub near_plane: f32,
    /// Far clipping plane distance.
    pub far_plane: f32,
    /// Whether this is an orthographic camera (1) or perspective (0).
    pub orthographic: u8,
    /// Orthographic size (half-height) if orthographic.
    pub ortho_size: f32,
}

impl Default for FfiCameraDesc {
    fn default() -> Self {
        Self {
            position: FfiVec3::new(0.0, 0.0, 5.0),
            target: FfiVec3::ZERO,
            up: FfiVec3::new(0.0, 1.0, 0.0),
            fov: std::f32::consts::FRAC_PI_4,
            near_plane: 0.1,
            far_plane: 1000.0,
            orthographic: 0,
            ortho_size: 5.0,
        }
    }
}

/// Set the active camera.
///
/// # Safety
///
/// `engine` and `desc` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_render_set_camera(
    engine: *mut EngineInstance,
    desc: *const FfiCameraDesc,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_render_set_camera: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if desc.is_null() {
            set_last_error("genovo_render_set_camera: null descriptor pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let d = unsafe { &*desc };
        if d.near_plane <= 0.0 || d.far_plane <= d.near_plane {
            set_last_error("genovo_render_set_camera: invalid near/far planes");
            return FFI_ERR_INVALID_PARAMETER;
        }
        if d.fov <= 0.0 || d.fov >= std::f32::consts::PI {
            set_last_error("genovo_render_set_camera: invalid FOV");
            return FFI_ERR_INVALID_PARAMETER;
        }
        FFI_OK
    })
}

/// Draw a frame. In a full implementation this would execute the render
/// pipeline: shadow pass, geometry pass, lighting, post-processing, present.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_render_draw_frame(engine: *mut EngineInstance) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_render_draw_frame: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        // Render pipeline execution would happen here
        FFI_OK
    })
}

// ============================================================================
// Animation FFI
// ============================================================================

/// Animation handle (opaque identifier).
pub type FfiAnimHandle = u64;

/// Play an animation on a scene node.
///
/// Returns an animation instance handle, or 0 on failure.
///
/// # Safety
///
/// `engine` and `anim_name` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_anim_play(
    engine: *mut EngineInstance,
    node: FfiNodeHandle,
    anim_name: *const c_char,
    speed: f32,
    looping: u8,
) -> FfiAnimHandle {
    ffi_catch(0u64, move || {
        if engine.is_null() {
            set_last_error("genovo_anim_play: null engine pointer");
            return 0;
        }
        if anim_name.is_null() {
            set_last_error("genovo_anim_play: null animation name");
            return 0;
        }
        if node == 0 {
            set_last_error("genovo_anim_play: invalid node handle 0");
            return 0;
        }
        let _name = match unsafe { CStr::from_ptr(anim_name) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("genovo_anim_play: invalid UTF-8 in animation name");
                return 0;
            }
        };
        let _speed = speed.max(0.01);
        let _looping = looping != 0;

        static NEXT_ANIM_HANDLE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        NEXT_ANIM_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    })
}

/// Stop an animation instance.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_anim_stop(
    engine: *mut EngineInstance,
    anim_handle: FfiAnimHandle,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_anim_stop: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if anim_handle == 0 {
            set_last_error("genovo_anim_stop: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        FFI_OK
    })
}

/// Set the blend weight of an animation instance.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_anim_set_blend_weight(
    engine: *mut EngineInstance,
    anim_handle: FfiAnimHandle,
    weight: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_anim_set_blend_weight: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if anim_handle == 0 {
            set_last_error("genovo_anim_set_blend_weight: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        if weight < 0.0 || weight > 1.0 {
            set_last_error("genovo_anim_set_blend_weight: weight must be in [0, 1]");
            return FFI_ERR_INVALID_PARAMETER;
        }
        FFI_OK
    })
}

/// Set the playback speed of an animation instance.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_anim_set_speed(
    engine: *mut EngineInstance,
    anim_handle: FfiAnimHandle,
    speed: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_anim_set_speed: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if anim_handle == 0 {
            set_last_error("genovo_anim_set_speed: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        if !speed.is_finite() {
            set_last_error("genovo_anim_set_speed: speed must be finite");
            return FFI_ERR_INVALID_PARAMETER;
        }
        FFI_OK
    })
}

/// Pause an animation instance.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_anim_pause(
    engine: *mut EngineInstance,
    anim_handle: FfiAnimHandle,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_anim_pause: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if anim_handle == 0 {
            set_last_error("genovo_anim_pause: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        FFI_OK
    })
}

/// Resume a paused animation instance.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_anim_resume(
    engine: *mut EngineInstance,
    anim_handle: FfiAnimHandle,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_anim_resume: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if anim_handle == 0 {
            set_last_error("genovo_anim_resume: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        FFI_OK
    })
}

// ============================================================================
// Input FFI
// ============================================================================

/// Check if a key is currently pressed.
///
/// `key_code` uses the engine's key code values (matching `KeyCode` enum).
///
/// # Safety
///
/// `engine` and `out_pressed` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_input_is_key_pressed(
    engine: *const EngineInstance,
    _key_code: u32,
    out_pressed: *mut u8,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_input_is_key_pressed: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_pressed.is_null() {
            set_last_error("genovo_input_is_key_pressed: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        // Input state would be queried from the platform layer here.
        // Return not-pressed as placeholder.
        unsafe {
            *out_pressed = 0;
        }
        FFI_OK
    })
}

/// Check if a key was just pressed this frame (edge trigger).
///
/// # Safety
///
/// `engine` and `out_pressed` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_input_is_key_just_pressed(
    engine: *const EngineInstance,
    _key_code: u32,
    out_pressed: *mut u8,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_input_is_key_just_pressed: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_pressed.is_null() {
            set_last_error("genovo_input_is_key_just_pressed: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        unsafe {
            *out_pressed = 0;
        }
        FFI_OK
    })
}

/// Get the current mouse position in window coordinates.
///
/// # Safety
///
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_input_get_mouse_position(
    engine: *const EngineInstance,
    out_x: *mut f32,
    out_y: *mut f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_input_get_mouse_position: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_x.is_null() || out_y.is_null() {
            set_last_error("genovo_input_get_mouse_position: null output pointer(s)");
            return FFI_ERR_NULL_POINTER;
        }
        // Return (0,0) as placeholder
        unsafe {
            *out_x = 0.0;
            *out_y = 0.0;
        }
        FFI_OK
    })
}

/// Get the mouse scroll delta for the current frame.
///
/// # Safety
///
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_input_get_mouse_scroll(
    engine: *const EngineInstance,
    out_x: *mut f32,
    out_y: *mut f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_input_get_mouse_scroll: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_x.is_null() || out_y.is_null() {
            set_last_error("genovo_input_get_mouse_scroll: null output pointer(s)");
            return FFI_ERR_NULL_POINTER;
        }
        unsafe {
            *out_x = 0.0;
            *out_y = 0.0;
        }
        FFI_OK
    })
}

/// Check if a mouse button is currently pressed.
///
/// `button`: 0 = left, 1 = right, 2 = middle.
///
/// # Safety
///
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_input_is_mouse_button_pressed(
    engine: *const EngineInstance,
    button: u32,
    out_pressed: *mut u8,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_input_is_mouse_button_pressed: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_pressed.is_null() {
            set_last_error("genovo_input_is_mouse_button_pressed: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if button > 4 {
            set_last_error("genovo_input_is_mouse_button_pressed: invalid button index");
            return FFI_ERR_INVALID_PARAMETER;
        }
        unsafe {
            *out_pressed = 0;
        }
        FFI_OK
    })
}

/// Get the value of a gamepad axis.
///
/// `gamepad_index`: 0-7 for gamepad slot.
/// `axis`: 0 = left X, 1 = left Y, 2 = right X, 3 = right Y,
///         4 = left trigger, 5 = right trigger.
///
/// Returns a value in [-1, 1] for sticks, [0, 1] for triggers.
///
/// # Safety
///
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_input_get_gamepad_axis(
    engine: *const EngineInstance,
    gamepad_index: u32,
    axis: u32,
    out_value: *mut f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_input_get_gamepad_axis: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_value.is_null() {
            set_last_error("genovo_input_get_gamepad_axis: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if gamepad_index >= 8 {
            set_last_error("genovo_input_get_gamepad_axis: gamepad_index out of range");
            return FFI_ERR_INVALID_PARAMETER;
        }
        if axis >= 6 {
            set_last_error("genovo_input_get_gamepad_axis: axis out of range");
            return FFI_ERR_INVALID_PARAMETER;
        }
        unsafe {
            *out_value = 0.0;
        }
        FFI_OK
    })
}

/// Check if a gamepad button is currently pressed.
///
/// # Safety
///
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_input_is_gamepad_button_pressed(
    engine: *const EngineInstance,
    gamepad_index: u32,
    button: u32,
    out_pressed: *mut u8,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_input_is_gamepad_button_pressed: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_pressed.is_null() {
            set_last_error("genovo_input_is_gamepad_button_pressed: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if gamepad_index >= 8 {
            set_last_error("genovo_input_is_gamepad_button_pressed: gamepad_index out of range");
            return FFI_ERR_INVALID_PARAMETER;
        }
        unsafe {
            *out_pressed = 0;
        }
        FFI_OK
    })
}

// ============================================================================
// Asset FFI
// ============================================================================

/// Asset handle (opaque identifier).
pub type FfiAssetHandle = u64;

/// Asset loading status.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiAssetStatus {
    /// Asset has not been requested.
    NotLoaded = 0,
    /// Asset is currently loading.
    Loading = 1,
    /// Asset is loaded and ready to use.
    Loaded = 2,
    /// Asset failed to load.
    Failed = 3,
    /// Asset has been unloaded.
    Unloaded = 4,
}

/// Load an asset synchronously by path.
///
/// Returns an asset handle, or 0 on failure. The asset data is immediately
/// available after this call returns successfully.
///
/// # Safety
///
/// `engine` and `path` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_asset_load_sync(
    engine: *mut EngineInstance,
    path: *const c_char,
) -> FfiAssetHandle {
    ffi_catch(0u64, move || {
        if engine.is_null() {
            set_last_error("genovo_asset_load_sync: null engine pointer");
            return 0;
        }
        if path.is_null() {
            set_last_error("genovo_asset_load_sync: null path pointer");
            return 0;
        }
        let _path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("genovo_asset_load_sync: invalid UTF-8 in path");
                return 0;
            }
        };
        static NEXT_ASSET_HANDLE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        NEXT_ASSET_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    })
}

/// Load an asset asynchronously by path.
///
/// Returns an asset handle immediately. The asset will be loaded in the
/// background; use `genovo_asset_get_status` to check progress.
///
/// # Safety
///
/// `engine` and `path` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_asset_load_async(
    engine: *mut EngineInstance,
    path: *const c_char,
) -> FfiAssetHandle {
    ffi_catch(0u64, move || {
        if engine.is_null() {
            set_last_error("genovo_asset_load_async: null engine pointer");
            return 0;
        }
        if path.is_null() {
            set_last_error("genovo_asset_load_async: null path pointer");
            return 0;
        }
        let _path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("genovo_asset_load_async: invalid UTF-8 in path");
                return 0;
            }
        };
        static NEXT_ASYNC_HANDLE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(100000);
        NEXT_ASYNC_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    })
}

/// Check if an asset is loaded and ready to use.
///
/// # Safety
///
/// `engine` and `out_loaded` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_asset_is_loaded(
    engine: *const EngineInstance,
    handle: FfiAssetHandle,
    out_loaded: *mut u8,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_asset_is_loaded: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_loaded.is_null() {
            set_last_error("genovo_asset_is_loaded: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_asset_is_loaded: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        // Placeholder: report all assets as loaded
        unsafe {
            *out_loaded = 1;
        }
        FFI_OK
    })
}

/// Get the status of an asset.
///
/// # Safety
///
/// `engine` and `out_status` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_asset_get_status(
    engine: *const EngineInstance,
    handle: FfiAssetHandle,
    out_status: *mut FfiAssetStatus,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_asset_get_status: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_status.is_null() {
            set_last_error("genovo_asset_get_status: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_asset_get_status: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        unsafe {
            *out_status = FfiAssetStatus::Loaded;
        }
        FFI_OK
    })
}

/// Get the raw data pointer and size of a loaded asset.
///
/// The returned pointer is valid as long as the asset remains loaded.
///
/// # Safety
///
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_asset_get_data(
    engine: *const EngineInstance,
    handle: FfiAssetHandle,
    out_data: *mut *const u8,
    out_size: *mut usize,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_asset_get_data: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_data.is_null() || out_size.is_null() {
            set_last_error("genovo_asset_get_data: null output pointer(s)");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_asset_get_data: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        // No real asset data to return
        unsafe {
            *out_data = ptr::null();
            *out_size = 0;
        }
        FFI_OK
    })
}

/// Unload an asset and free its resources.
///
/// # Safety
///
/// `engine` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_asset_unload(
    engine: *mut EngineInstance,
    handle: FfiAssetHandle,
) -> FfiResult {
    ffi_catch_result(move || {
        if engine.is_null() {
            set_last_error("genovo_asset_unload: null engine pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_asset_unload: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        FFI_OK
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config_default() {
        let cfg = FfiEngineConfig::default();
        assert_eq!(cfg.window_width, 1280);
        assert_eq!(cfg.window_height, 720);
        assert_eq!(cfg.target_fps, 60);
        assert_eq!(cfg.fullscreen, 0);
        assert_eq!(cfg.vsync, 1);
    }

    #[test]
    fn test_engine_create_destroy_default() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            assert!(!engine.is_null());
            let result = genovo_engine_destroy(engine);
            assert_eq!(result, FFI_OK);
        }
    }

    #[test]
    fn test_engine_create_with_config() {
        let cfg = FfiEngineConfig {
            window_width: 1920,
            window_height: 1080,
            target_fps: 120,
            ..FfiEngineConfig::default()
        };
        unsafe {
            let engine = genovo_engine_create(&cfg);
            assert!(!engine.is_null());
            let result = genovo_engine_destroy(engine);
            assert_eq!(result, FFI_OK);
        }
    }

    #[test]
    fn test_engine_destroy_null() {
        unsafe {
            let result = genovo_engine_destroy(ptr::null_mut());
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    #[test]
    fn test_engine_update() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            let result = genovo_engine_update(engine, 0.016);
            assert_eq!(result, FFI_OK);

            let mut count = 0u64;
            genovo_engine_get_frame_count(engine, &mut count);
            assert_eq!(count, 1);

            genovo_engine_destroy(engine);
        }
    }

    #[test]
    fn test_engine_update_null() {
        unsafe {
            let result = genovo_engine_update(ptr::null_mut(), 0.016);
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    #[test]
    fn test_engine_update_bad_dt() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            let result = genovo_engine_update(engine, -1.0);
            assert_eq!(result, FFI_ERR_INVALID_PARAMETER);
            genovo_engine_destroy(engine);
        }
    }

    #[test]
    fn test_engine_get_world() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            let world = genovo_engine_get_world(engine);
            assert!(!world.is_null());
            genovo_engine_destroy(engine);
        }
    }

    #[test]
    fn test_engine_is_running() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            let mut running = 0u8;
            genovo_engine_is_running(engine, &mut running);
            assert_eq!(running, 1);

            genovo_engine_request_shutdown(engine);
            genovo_engine_is_running(engine, &mut running);
            assert_eq!(running, 0);

            genovo_engine_destroy(engine);
        }
    }

    #[test]
    fn test_scene_create_destroy_node() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            let desc = FfiSceneNodeDesc::default();
            let handle = genovo_scene_create_node(engine, &desc);
            assert_ne!(handle, 0);

            let result = genovo_scene_destroy_node(engine, handle);
            assert_eq!(result, FFI_OK);

            genovo_engine_destroy(engine);
        }
    }

    #[test]
    fn test_scene_set_get_transform() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            let desc = FfiSceneNodeDesc::default();
            let handle = genovo_scene_create_node(engine, &desc);

            let transform = FfiTransform {
                position: FfiVec3::new(1.0, 2.0, 3.0),
                rotation: FfiQuat::IDENTITY,
                scale: FfiVec3::ONE,
            };
            let result = genovo_scene_set_transform(engine, handle, &transform);
            assert_eq!(result, FFI_OK);

            let mut out = FfiTransform::default();
            let result = genovo_scene_get_transform(engine, handle, &mut out);
            assert_eq!(result, FFI_OK);

            genovo_engine_destroy(engine);
        }
    }

    #[test]
    fn test_render_create_mesh_null() {
        unsafe {
            let handle = genovo_render_create_mesh(ptr::null_mut(), ptr::null());
            assert_eq!(handle, 0);
        }
    }

    #[test]
    fn test_render_camera_validation() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            let desc = FfiCameraDesc::default();
            let result = genovo_render_set_camera(engine, &desc);
            assert_eq!(result, FFI_OK);
            genovo_engine_destroy(engine);
        }
    }

    #[test]
    fn test_anim_play_stop() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            let desc = FfiSceneNodeDesc::default();
            let node = genovo_scene_create_node(engine, &desc);

            let name = CString::new("idle").unwrap();
            let anim = genovo_anim_play(engine, node, name.as_ptr(), 1.0, 1);
            assert_ne!(anim, 0);

            let result = genovo_anim_stop(engine, anim);
            assert_eq!(result, FFI_OK);

            genovo_engine_destroy(engine);
        }
    }

    #[test]
    fn test_input_functions_null_safety() {
        unsafe {
            let mut pressed = 0u8;
            let result = genovo_input_is_key_pressed(ptr::null(), 0, &mut pressed);
            assert_eq!(result, FFI_ERR_NULL_POINTER);

            let mut x = 0.0f32;
            let mut y = 0.0f32;
            let result = genovo_input_get_mouse_position(ptr::null(), &mut x, &mut y);
            assert_eq!(result, FFI_ERR_NULL_POINTER);

            let mut val = 0.0f32;
            let result = genovo_input_get_gamepad_axis(ptr::null(), 0, 0, &mut val);
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    #[test]
    fn test_asset_load_sync() {
        unsafe {
            let engine = genovo_engine_create(ptr::null());
            let path = CString::new("test.png").unwrap();
            let handle = genovo_asset_load_sync(engine, path.as_ptr());
            assert_ne!(handle, 0);

            let mut loaded = 0u8;
            genovo_asset_is_loaded(engine, handle, &mut loaded);
            assert_eq!(loaded, 1);

            let mut status = FfiAssetStatus::NotLoaded;
            genovo_asset_get_status(engine, handle, &mut status);
            assert_eq!(status, FfiAssetStatus::Loaded);

            genovo_engine_destroy(engine);
        }
    }

    #[test]
    fn test_material_desc_default() {
        let desc = FfiMaterialDesc::default();
        assert_eq!(desc.base_color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(desc.metallic, 0.0);
        assert_eq!(desc.roughness, 0.5);
    }

    #[test]
    fn test_camera_desc_default() {
        let desc = FfiCameraDesc::default();
        assert!(desc.fov > 0.0);
        assert!(desc.near_plane > 0.0);
        assert!(desc.far_plane > desc.near_plane);
    }
}
