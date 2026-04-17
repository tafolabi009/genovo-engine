//! # Genovo FFI Layer
//!
//! Provides a C-compatible foreign function interface (FFI) for the Genovo
//! engine, enabling interoperability with C/C++ consumers.
//!
//! All public functions use `extern "C"` linkage and `#[unsafe(no_mangle)]` to produce
//! stable symbol names. Data is exchanged through `#[repr(C)]` structs.
//!
//! ## Safety Contract
//!
//! Every exported function performs:
//! - Null-pointer validation on all pointer arguments
//! - `catch_unwind` to prevent Rust panics from crossing the FFI boundary
//! - Error reporting via thread-local last-error string
//!
//! Callers must eventually destroy every object they create (e.g.
//! `genovo_physics_world_destroy` for worlds created with
//! `genovo_physics_world_create`).

pub mod audio_bridge;
pub mod bindings_gen;
pub mod cpp_bridge;
pub mod engine_ffi;
pub mod lua_bridge;
pub mod physx_bridge;

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;
use std::panic;
use std::ptr;

use genovo_audio::{AudioClip, AudioMixer, MixerHandle, SoftwareMixer, AudioSource, ChannelState};
use genovo_core::math::{Quat, Vec3};
use genovo_core::Transform;
use genovo_physics::{
    BodyType, ColliderDesc, CollisionShape, PhysicsMaterial, PhysicsWorld, RaycastHit,
    RigidBodyDesc, RigidBodyHandle,
};

// ============================================================================
// Error codes (FfiResult)
// ============================================================================

/// Integer result code returned by all FFI functions.
///
/// Zero means success; negative values indicate specific error categories.
/// When a non-zero code is returned the caller should inspect the thread-local
/// error string via `genovo_get_last_error()` for a human-readable message.
pub const FFI_OK: i32 = 0;
pub const FFI_ERR_NULL_POINTER: i32 = -1;
pub const FFI_ERR_INVALID_HANDLE: i32 = -2;
pub const FFI_ERR_INTERNAL: i32 = -3;
pub const FFI_ERR_INVALID_PARAMETER: i32 = -4;
pub const FFI_ERR_OUT_OF_MEMORY: i32 = -5;
pub const FFI_ERR_NOT_IMPLEMENTED: i32 = -6;
pub const FFI_ERR_PANIC: i32 = -7;
pub const FFI_ERR_AUDIO_DECODE: i32 = -8;
pub const FFI_ERR_AUDIO_PLAYBACK: i32 = -9;
pub const FFI_ERR_PHYSICS_BODY: i32 = -10;

/// Type alias used by all FFI functions.  `0` is success, negative is error.
pub type FfiResult = i32;

// ============================================================================
// Thread-local last error
// ============================================================================

thread_local! {
    /// Stores the last error message as a C-compatible string.
    /// Kept per-thread to avoid contention and match typical C error patterns.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Record an error message into the thread-local slot.
fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

/// Record an error message from a formatted string.
fn set_last_error_fmt(args: std::fmt::Arguments<'_>) {
    set_last_error(&args.to_string());
}

/// Retrieve the last error message set on the current thread.
///
/// Returns a null pointer if no error has been recorded since the last call to
/// `genovo_clear_error()` (or if no error has ever been set).
///
/// # Safety
///
/// The returned pointer is valid until the next FFI call on the **same thread**
/// that modifies the error state. Callers must copy the string if they need to
/// keep it beyond that point.
#[unsafe(no_mangle)]
pub extern "C" fn genovo_get_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(cstr) => cstr.as_ptr(),
            None => ptr::null(),
        }
    })
}

/// Clear the thread-local error state.
#[unsafe(no_mangle)]
pub extern "C" fn genovo_clear_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// ============================================================================
// Helper: catch_unwind wrapper
// ============================================================================

/// Runs `f` inside `catch_unwind`. On panic the thread-local error is set and
/// `err_val` is returned.
fn ffi_catch<F, T>(err_val: T, f: F) -> T
where
    F: FnOnce() -> T,
{
    match panic::catch_unwind(panic::AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            set_last_error("Rust panic caught at FFI boundary");
            err_val
        }
    }
}

/// Variant that returns an `FfiResult` error code on panic.
fn ffi_catch_result<F>(f: F) -> FfiResult
where
    F: FnOnce() -> FfiResult,
{
    ffi_catch(FFI_ERR_PANIC, f)
}

// ============================================================================
// #[repr(C)] data types
// ============================================================================

/// 3-component floating-point vector in C-compatible layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FfiVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl FfiVec3 {
    /// Construct a new `FfiVec3`.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// The unit vector (1, 1, 1).
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
}

impl From<Vec3> for FfiVec3 {
    fn from(v: Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<FfiVec3> for Vec3 {
    fn from(v: FfiVec3) -> Self {
        Vec3::new(v.x, v.y, v.z)
    }
}

/// Quaternion in C-compatible layout (x, y, z, w order).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfiQuat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl FfiQuat {
    /// Construct a new `FfiQuat`.
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// The identity quaternion (no rotation).
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };
}

impl Default for FfiQuat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Quat> for FfiQuat {
    fn from(q: Quat) -> Self {
        Self {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }
}

impl From<FfiQuat> for Quat {
    fn from(q: FfiQuat) -> Self {
        Quat::from_xyzw(q.x, q.y, q.z, q.w)
    }
}

/// Transform (position + rotation + scale) in C-compatible layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfiTransform {
    pub position: FfiVec3,
    pub rotation: FfiQuat,
    pub scale: FfiVec3,
}

impl FfiTransform {
    /// An identity transform: origin position, no rotation, unit scale.
    pub const IDENTITY: Self = Self {
        position: FfiVec3::ZERO,
        rotation: FfiQuat::IDENTITY,
        scale: FfiVec3::ONE,
    };
}

impl Default for FfiTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Transform> for FfiTransform {
    fn from(t: Transform) -> Self {
        Self {
            position: FfiVec3::from(t.position),
            rotation: FfiQuat::from(t.rotation),
            scale: FfiVec3::from(t.scale),
        }
    }
}

impl From<FfiTransform> for Transform {
    fn from(t: FfiTransform) -> Self {
        Transform {
            position: Vec3::from(t.position),
            rotation: Quat::from(t.rotation),
            scale: Vec3::from(t.scale),
        }
    }
}

/// Rigid body descriptor passed across the FFI boundary.
///
/// `body_type` encoding: 0 = Static, 1 = Dynamic, 2 = Kinematic.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiRigidBodyDesc {
    /// 0 = Static, 1 = Dynamic, 2 = Kinematic.
    pub body_type: u8,
    /// Mass in kilograms (ignored for static bodies).
    pub mass: f32,
    /// Coulomb friction coefficient [0, 1].
    pub friction: f32,
    /// Coefficient of restitution (bounciness) [0, 1].
    pub restitution: f32,
    /// Linear damping factor.
    pub linear_damping: f32,
    /// Angular damping factor.
    pub angular_damping: f32,
    /// Initial position.
    pub position: FfiVec3,
    /// Initial rotation.
    pub rotation: FfiQuat,
    /// Whether the body starts awake (1) or sleeping (0).
    pub is_awake: u8,
    /// Whether continuous collision detection is enabled.
    pub ccd_enabled: u8,
    /// Padding / reserved for future use.
    pub _reserved: [u8; 2],
}

impl Default for FfiRigidBodyDesc {
    fn default() -> Self {
        Self {
            body_type: 1, // Dynamic
            mass: 1.0,
            friction: 0.5,
            restitution: 0.3,
            linear_damping: 0.0,
            angular_damping: 0.05,
            position: FfiVec3::ZERO,
            rotation: FfiQuat::IDENTITY,
            is_awake: 1,
            ccd_enabled: 0,
            _reserved: [0; 2],
        }
    }
}

/// Convert the FFI body-type byte to the engine enum.
fn body_type_from_u8(v: u8) -> Option<BodyType> {
    match v {
        0 => Some(BodyType::Static),
        1 => Some(BodyType::Dynamic),
        2 => Some(BodyType::Kinematic),
        _ => None,
    }
}

/// Build a `RigidBodyDesc` from the C-side descriptor.
fn rigid_body_desc_from_ffi(ffi: &FfiRigidBodyDesc) -> Option<RigidBodyDesc> {
    let bt = body_type_from_u8(ffi.body_type)?;
    Some(RigidBodyDesc {
        body_type: bt,
        mass: ffi.mass,
        position: Vec3::from(ffi.position),
        rotation: Quat::from(ffi.rotation),
        friction: ffi.friction,
        restitution: ffi.restitution,
        linear_damping: ffi.linear_damping,
        angular_damping: ffi.angular_damping,
    })
}

/// Result of a physics raycast, returned to C callers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiRaycastResult {
    /// World-space hit position.
    pub point: FfiVec3,
    /// Surface normal at the hit point.
    pub normal: FfiVec3,
    /// Distance from the ray origin to the hit point.
    pub distance: f32,
    /// Handle of the rigid body that was hit.  `0` typically means invalid.
    pub body_handle: u64,
}

impl Default for FfiRaycastResult {
    fn default() -> Self {
        Self {
            point: FfiVec3::ZERO,
            normal: FfiVec3::ZERO,
            distance: 0.0,
            body_handle: 0,
        }
    }
}

impl From<RaycastHit> for FfiRaycastResult {
    fn from(hit: RaycastHit) -> Self {
        Self {
            point: FfiVec3::from(hit.point),
            normal: FfiVec3::from(hit.normal),
            distance: hit.distance,
            body_handle: hit.body.0,
        }
    }
}

// ============================================================================
// Physics FFI
// ============================================================================

/// Create a new physics world with the specified gravity vector.
///
/// Returns a heap-allocated `PhysicsWorld` as an opaque pointer. The caller
/// **must** eventually free it with `genovo_physics_world_destroy`.
///
/// Returns null on failure (check `genovo_get_last_error`).
///
/// # Safety
///
/// Caller owns the returned pointer and must not use it after calling
/// `genovo_physics_world_destroy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_world_create(
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
) -> *mut PhysicsWorld {
    ffi_catch(ptr::null_mut(), || {
        let gravity = Vec3::new(gravity_x, gravity_y, gravity_z);
        let world = PhysicsWorld::new(gravity);
        let boxed = Box::new(world);
        Box::into_raw(boxed)
    })
}

/// Destroy a physics world previously created with `genovo_physics_world_create`.
///
/// After this call the pointer is invalid and must not be used.
///
/// # Safety
///
/// `world` must be a valid pointer returned by `genovo_physics_world_create`
/// and must not have been previously destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_world_destroy(world: *mut PhysicsWorld) -> FfiResult {
    ffi_catch_result(|| {
        if world.is_null() {
            set_last_error("genovo_physics_world_destroy: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        // Reconstruct the Box so Rust drops it properly.
        unsafe {
            let _ = Box::from_raw(world);
        }
        FFI_OK
    })
}

/// Advance the physics simulation by `dt` seconds.
///
/// # Safety
///
/// `world` must be a valid, non-null pointer to a live `PhysicsWorld`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_world_step(
    world: *mut PhysicsWorld,
    dt: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_world_step: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if dt < 0.0 || !dt.is_finite() {
            set_last_error("genovo_physics_world_step: dt must be non-negative and finite");
            return FFI_ERR_INVALID_PARAMETER;
        }
        let w = unsafe { &mut *world };
        match w.step(dt) {
            Ok(()) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_physics_world_step: {}", e));
                FFI_ERR_INTERNAL
            }
        }
    })
}

/// Set the gravity vector on an existing physics world.
///
/// # Safety
///
/// `world` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_world_set_gravity(
    world: *mut PhysicsWorld,
    x: f32,
    y: f32,
    z: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_world_set_gravity: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let w = unsafe { &mut *world };
        w.set_gravity(Vec3::new(x, y, z));
        FFI_OK
    })
}

/// Add a rigid body to the physics world.
///
/// `desc` must point to a valid `FfiRigidBodyDesc`.
///
/// Returns the body handle as a `u64`. On failure returns `0` and sets the
/// error string.
///
/// # Safety
///
/// Both `world` and `desc` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_add_body(
    world: *mut PhysicsWorld,
    desc: *const FfiRigidBodyDesc,
) -> u64 {
    ffi_catch(0u64, move || {
        if world.is_null() {
            set_last_error("genovo_physics_add_body: null world pointer");
            return 0;
        }
        if desc.is_null() {
            set_last_error("genovo_physics_add_body: null descriptor pointer");
            return 0;
        }
        let ffi_desc = unsafe { &*desc };
        let rb_desc = match rigid_body_desc_from_ffi(ffi_desc) {
            Some(d) => d,
            None => {
                set_last_error_fmt(format_args!(
                    "genovo_physics_add_body: invalid body_type {}",
                    ffi_desc.body_type
                ));
                return 0;
            }
        };
        let w = unsafe { &mut *world };
        match w.add_body(&rb_desc) {
            Ok(handle) => handle.0,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_physics_add_body: {}", e));
                0
            }
        }
    })
}

/// Remove a rigid body from the physics world by handle.
///
/// # Safety
///
/// `world` must be a valid, non-null pointer. `handle` must be a value
/// previously returned by `genovo_physics_add_body`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_remove_body(
    world: *mut PhysicsWorld,
    handle: u64,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_remove_body: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_physics_remove_body: handle 0 is reserved / invalid");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &mut *world };
        match w.remove_body(RigidBodyHandle(handle)) {
            Ok(()) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_physics_remove_body: {}", e));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Set the world-space position of a rigid body.
///
/// # Safety
///
/// `world` must be valid and non-null. `handle` must refer to an existing body.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_set_position(
    world: *mut PhysicsWorld,
    handle: u64,
    x: f32,
    y: f32,
    z: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_set_position: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_physics_set_position: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &mut *world };
        match w.get_body_mut(RigidBodyHandle(handle)) {
            Ok(body) => {
                body.position = Vec3::new(x, y, z);
                FFI_OK
            }
            Err(e) => {
                set_last_error_fmt(format_args!(
                    "genovo_physics_set_position: {}",
                    e
                ));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Read the world-space position of a rigid body into caller-supplied floats.
///
/// `out_x`, `out_y`, `out_z` must each point to a writable `f32`.
///
/// # Safety
///
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_get_position(
    world: *mut PhysicsWorld,
    handle: u64,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_get_position: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_x.is_null() || out_y.is_null() || out_z.is_null() {
            set_last_error("genovo_physics_get_position: null output pointer(s)");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_physics_get_position: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &*world };
        match w.get_body(RigidBodyHandle(handle)) {
            Ok(body) => {
                unsafe {
                    *out_x = body.position.x;
                    *out_y = body.position.y;
                    *out_z = body.position.z;
                }
                FFI_OK
            }
            Err(e) => {
                set_last_error_fmt(format_args!(
                    "genovo_physics_get_position: {}",
                    e
                ));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Apply a force (in world coordinates) to a rigid body.
///
/// The force is applied at the center of mass for the current simulation step.
///
/// # Safety
///
/// `world` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_add_force(
    world: *mut PhysicsWorld,
    handle: u64,
    fx: f32,
    fy: f32,
    fz: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_add_force: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_physics_add_force: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &mut *world };
        match w.get_body_mut(RigidBodyHandle(handle)) {
            Ok(body) => {
                body.apply_force(Vec3::new(fx, fy, fz), genovo_physics::ForceMode::Force);
                FFI_OK
            }
            Err(e) => {
                set_last_error_fmt(format_args!(
                    "genovo_physics_add_force: {}",
                    e
                ));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Cast a ray into the physics world and write up to `max_results` hits into
/// the caller-provided buffer.
///
/// Returns the number of hits written. On error, returns `0` and sets the
/// error string.
///
/// # Parameters
///
/// - `ox, oy, oz` -- ray origin in world space.
/// - `dx, dy, dz` -- ray direction (will be normalized internally).
/// - `max_dist`    -- maximum ray travel distance.
/// - `out_results` -- pointer to an array of `FfiRaycastResult` with room for
///                    at least `max_results` entries.
/// - `max_results` -- capacity of `out_results`.
///
/// # Safety
///
/// `world` and `out_results` must be valid, non-null pointers. `out_results`
/// must have room for `max_results` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_raycast(
    world: *mut PhysicsWorld,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    max_dist: f32,
    out_results: *mut FfiRaycastResult,
    max_results: u32,
) -> u32 {
    ffi_catch(0u32, move || {
        if world.is_null() {
            set_last_error("genovo_physics_raycast: null world pointer");
            return 0;
        }
        if out_results.is_null() {
            set_last_error("genovo_physics_raycast: null output pointer");
            return 0;
        }
        if max_results == 0 {
            return 0;
        }

        let origin = Vec3::new(ox, oy, oz);
        let direction = Vec3::new(dx, dy, dz);

        // Validate direction is not zero-length.
        let len_sq = dx * dx + dy * dy + dz * dz;
        if len_sq < 1e-12 {
            set_last_error("genovo_physics_raycast: direction vector is zero");
            return 0;
        }

        let w = unsafe { &*world };
        let hits = w.raycast(origin, direction, max_dist);

        let count = hits.len().min(max_results as usize);
        for i in 0..count {
            unsafe {
                let dest = out_results.add(i);
                *dest = FfiRaycastResult::from(hits[i].clone());
            }
        }

        count as u32
    })
}

/// Add a box collider to a rigid body.
///
/// `half_x`, `half_y`, `half_z` are the half-extents of the box shape.
///
/// Returns `0` on success, negative error code on failure.
///
/// # Safety
///
/// `world` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_add_box_collider(
    world: *mut PhysicsWorld,
    body_handle: u64,
    half_x: f32,
    half_y: f32,
    half_z: f32,
    friction: f32,
    restitution: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_add_box_collider: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if body_handle == 0 {
            set_last_error("genovo_physics_add_box_collider: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &mut *world };
        let desc = ColliderDesc {
            shape: CollisionShape::Box {
                half_extents: Vec3::new(half_x, half_y, half_z),
            },
            material: PhysicsMaterial {
                friction,
                restitution,
                density: 1000.0,
            },
            ..Default::default()
        };
        match w.add_collider(RigidBodyHandle(body_handle), &desc) {
            Ok(_) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_physics_add_box_collider: {}", e));
                FFI_ERR_INTERNAL
            }
        }
    })
}

/// Add a sphere collider to a rigid body.
///
/// # Safety
///
/// `world` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_add_sphere_collider(
    world: *mut PhysicsWorld,
    body_handle: u64,
    radius: f32,
    friction: f32,
    restitution: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_add_sphere_collider: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if body_handle == 0 {
            set_last_error("genovo_physics_add_sphere_collider: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        if radius <= 0.0 {
            set_last_error("genovo_physics_add_sphere_collider: radius must be positive");
            return FFI_ERR_INVALID_PARAMETER;
        }
        let w = unsafe { &mut *world };
        let desc = ColliderDesc {
            shape: CollisionShape::Sphere { radius },
            material: PhysicsMaterial {
                friction,
                restitution,
                density: 1000.0,
            },
            ..Default::default()
        };
        match w.add_collider(RigidBodyHandle(body_handle), &desc) {
            Ok(_) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_physics_add_sphere_collider: {}", e));
                FFI_ERR_INTERNAL
            }
        }
    })
}

/// Add a capsule collider to a rigid body.
///
/// # Safety
///
/// `world` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_add_capsule_collider(
    world: *mut PhysicsWorld,
    body_handle: u64,
    radius: f32,
    half_height: f32,
    friction: f32,
    restitution: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_add_capsule_collider: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if body_handle == 0 {
            set_last_error("genovo_physics_add_capsule_collider: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        if radius <= 0.0 || half_height <= 0.0 {
            set_last_error(
                "genovo_physics_add_capsule_collider: radius and half_height must be positive",
            );
            return FFI_ERR_INVALID_PARAMETER;
        }
        let w = unsafe { &mut *world };
        let desc = ColliderDesc {
            shape: CollisionShape::Capsule {
                radius,
                half_height,
            },
            material: PhysicsMaterial {
                friction,
                restitution,
                density: 1000.0,
            },
            ..Default::default()
        };
        match w.add_collider(RigidBodyHandle(body_handle), &desc) {
            Ok(_) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_physics_add_capsule_collider: {}", e));
                FFI_ERR_INTERNAL
            }
        }
    })
}

/// Get the linear velocity of a rigid body.
///
/// # Safety
///
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_get_velocity(
    world: *mut PhysicsWorld,
    handle: u64,
    out_vx: *mut f32,
    out_vy: *mut f32,
    out_vz: *mut f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_get_velocity: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_vx.is_null() || out_vy.is_null() || out_vz.is_null() {
            set_last_error("genovo_physics_get_velocity: null output pointer(s)");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_physics_get_velocity: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &*world };
        match w.get_body(RigidBodyHandle(handle)) {
            Ok(body) => {
                let vel = body.linear_velocity;
                unsafe {
                    *out_vx = vel.x;
                    *out_vy = vel.y;
                    *out_vz = vel.z;
                }
                FFI_OK
            }
            Err(e) => {
                set_last_error_fmt(format_args!(
                    "genovo_physics_get_velocity: {}",
                    e
                ));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Set the linear velocity of a rigid body.
///
/// # Safety
///
/// `world` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_set_velocity(
    world: *mut PhysicsWorld,
    handle: u64,
    vx: f32,
    vy: f32,
    vz: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_set_velocity: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_physics_set_velocity: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &mut *world };
        match w.get_body_mut(RigidBodyHandle(handle)) {
            Ok(body) => {
                body.linear_velocity = Vec3::new(vx, vy, vz);
                FFI_OK
            }
            Err(e) => {
                set_last_error_fmt(format_args!(
                    "genovo_physics_set_velocity: {}",
                    e
                ));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Get the rotation (as a quaternion) of a rigid body.
///
/// # Safety
///
/// All pointer arguments must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_get_rotation(
    world: *mut PhysicsWorld,
    handle: u64,
    out_qx: *mut f32,
    out_qy: *mut f32,
    out_qz: *mut f32,
    out_qw: *mut f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_get_rotation: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_qx.is_null() || out_qy.is_null() || out_qz.is_null() || out_qw.is_null() {
            set_last_error("genovo_physics_get_rotation: null output pointer(s)");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_physics_get_rotation: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &*world };
        match w.get_body(RigidBodyHandle(handle)) {
            Ok(body) => {
                let rot = body.rotation;
                unsafe {
                    *out_qx = rot.x;
                    *out_qy = rot.y;
                    *out_qz = rot.z;
                    *out_qw = rot.w;
                }
                FFI_OK
            }
            Err(e) => {
                set_last_error_fmt(format_args!(
                    "genovo_physics_get_rotation: {}",
                    e
                ));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Set the rotation (as a quaternion) of a rigid body.
///
/// # Safety
///
/// `world` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_set_rotation(
    world: *mut PhysicsWorld,
    handle: u64,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_set_rotation: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_physics_set_rotation: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &mut *world };
        match w.get_body_mut(RigidBodyHandle(handle)) {
            Ok(body) => {
                body.rotation = Quat::from_xyzw(qx, qy, qz, qw);
                FFI_OK
            }
            Err(e) => {
                set_last_error_fmt(format_args!(
                    "genovo_physics_set_rotation: {}",
                    e
                ));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Get the full transform (position + rotation) of a rigid body.
///
/// # Safety
///
/// `world` and `out_transform` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_get_transform(
    world: *mut PhysicsWorld,
    handle: u64,
    out_transform: *mut FfiTransform,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_get_transform: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_transform.is_null() {
            set_last_error("genovo_physics_get_transform: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_physics_get_transform: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let w = unsafe { &*world };
        match w.get_body(RigidBodyHandle(handle)) {
            Ok(body) => {
                unsafe {
                    (*out_transform).position = FfiVec3::from(body.position);
                    (*out_transform).rotation = FfiQuat::from(body.rotation);
                    (*out_transform).scale = FfiVec3::ONE;
                }
                FFI_OK
            }
            Err(e) => {
                set_last_error_fmt(format_args!(
                    "genovo_physics_get_transform: {}",
                    e
                ));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Query the number of rigid bodies currently in the world.
///
/// # Safety
///
/// `world` must be valid and non-null. `out_count` must point to writable
/// memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_physics_body_count(
    world: *mut PhysicsWorld,
    out_count: *mut u32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_physics_body_count: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_count.is_null() {
            set_last_error("genovo_physics_body_count: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let w = unsafe { &*world };
        unsafe {
            *out_count = w.body_count() as u32;
        }
        FFI_OK
    })
}

// ============================================================================
// Audio FFI
// ============================================================================

/// Create a new software audio mixer.
///
/// Returns a heap-allocated `SoftwareMixer` as an opaque pointer. The caller
/// **must** eventually free it with `genovo_audio_mixer_destroy`.
///
/// Returns null on failure.
///
/// # Safety
///
/// Caller owns the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_mixer_create() -> *mut SoftwareMixer {
    ffi_catch(ptr::null_mut(), || {
        let mixer = SoftwareMixer::new(48000, 2, 1024, 64);
        let boxed = Box::new(mixer);
        Box::into_raw(boxed)
    })
}

/// Destroy a software audio mixer previously created with
/// `genovo_audio_mixer_create`.
///
/// # Safety
///
/// `mixer` must be a valid pointer returned by `genovo_audio_mixer_create` and
/// must not have been previously destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_mixer_destroy(mixer: *mut SoftwareMixer) -> FfiResult {
    ffi_catch_result(|| {
        if mixer.is_null() {
            set_last_error("genovo_audio_mixer_destroy: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        unsafe {
            let _ = Box::from_raw(mixer);
        }
        FFI_OK
    })
}

/// Update the audio mixer, advancing internal state by `dt` seconds.
///
/// This should be called once per frame (or at the audio-thread tick rate).
///
/// # Safety
///
/// `mixer` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_mixer_update(
    mixer: *mut SoftwareMixer,
    dt: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_mixer_update: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if dt < 0.0 || !dt.is_finite() {
            set_last_error("genovo_audio_mixer_update: dt must be non-negative and finite");
            return FFI_ERR_INVALID_PARAMETER;
        }
        let m = unsafe { &mut *mixer };
        m.update(dt);
        FFI_OK
    })
}

/// Load an audio clip from raw WAV data in memory.
///
/// `data` must point to `len` bytes of valid WAV-encoded audio. The data is
/// copied internally; the caller may free the original buffer after this call
/// returns.
///
/// Returns a heap-allocated `AudioClip` pointer, or null on failure.
///
/// # Safety
///
/// `data` must be a valid pointer to at least `len` bytes of readable memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_clip_from_wav(
    data: *const u8,
    len: usize,
) -> *mut AudioClip {
    ffi_catch(ptr::null_mut(), move || {
        if data.is_null() {
            set_last_error("genovo_audio_clip_from_wav: null data pointer");
            return ptr::null_mut();
        }
        if len == 0 {
            set_last_error("genovo_audio_clip_from_wav: zero-length data");
            return ptr::null_mut();
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        match AudioClip::from_wav_bytes(slice) {
            Ok(clip) => Box::into_raw(Box::new(clip)),
            Err(e) => {
                set_last_error_fmt(format_args!(
                    "genovo_audio_clip_from_wav: decode error: {}",
                    e
                ));
                ptr::null_mut()
            }
        }
    })
}

/// Destroy an audio clip previously created with `genovo_audio_clip_from_wav`.
///
/// # Safety
///
/// `clip` must be a valid pointer returned by `genovo_audio_clip_from_wav` and
/// must not have been previously destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_clip_destroy(clip: *mut AudioClip) -> FfiResult {
    ffi_catch_result(|| {
        if clip.is_null() {
            set_last_error("genovo_audio_clip_destroy: null clip pointer");
            return FFI_ERR_NULL_POINTER;
        }
        unsafe {
            let _ = Box::from_raw(clip);
        }
        FFI_OK
    })
}

/// Play an audio clip through the mixer.
///
/// # Parameters
///
/// - `mixer`   -- pointer to the software mixer.
/// - `clip`    -- pointer to the audio clip to play.
/// - `volume`  -- playback volume, where `1.0` is full volume.
/// - `pitch`   -- playback pitch multiplier (`1.0` = normal speed).
/// - `looping` -- `1` to loop, `0` for one-shot.
///
/// Returns a `u64` mixer handle that can be used with `genovo_audio_stop` and
/// `genovo_audio_set_volume`. Returns `0` on failure.
///
/// # Safety
///
/// `mixer` and `clip` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_play(
    mixer: *mut SoftwareMixer,
    clip: *const AudioClip,
    volume: f32,
    pitch: f32,
    looping: u8,
) -> u64 {
    ffi_catch(0u64, move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_play: null mixer pointer");
            return 0;
        }
        if clip.is_null() {
            set_last_error("genovo_audio_play: null clip pointer");
            return 0;
        }
        let m = unsafe { &mut *mixer };
        let c = unsafe { &*clip };
        // Load the clip if not already loaded, then play via AudioSource
        let clip_name = c.name.clone();
        // Ensure clip is loaded in the mixer
        if m.get_clip(&clip_name).is_none() {
            m.load_clip(c.clone());
        }
        let source = AudioSource {
            clip_name,
            volume: volume.clamp(0.0, 10.0),
            pitch: pitch.max(0.01),
            looping: looping != 0,
            ..Default::default()
        };
        match m.play(source) {
            Ok(handle) => handle.0,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_audio_play: {}", e));
                0
            }
        }
    })
}

/// Stop a sound that is currently playing.
///
/// # Safety
///
/// `mixer` must be valid and non-null. `handle` must have been returned by a
/// previous call to `genovo_audio_play`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_stop(
    mixer: *mut SoftwareMixer,
    handle: u64,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_stop: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_audio_stop: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let m = unsafe { &mut *mixer };
        match m.stop(MixerHandle(handle), 0.0) {
            Ok(()) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_audio_stop: {}", e));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Set the volume of a currently-playing sound.
///
/// # Safety
///
/// `mixer` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_set_volume(
    mixer: *mut SoftwareMixer,
    handle: u64,
    volume: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_set_volume: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_audio_set_volume: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let m = unsafe { &mut *mixer };
        match m.set_volume(MixerHandle(handle), volume.clamp(0.0, 10.0)) {
            Ok(()) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_audio_set_volume: {}", e));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Set the pitch of a currently-playing sound.
///
/// # Safety
///
/// `mixer` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_set_pitch(
    mixer: *mut SoftwareMixer,
    handle: u64,
    pitch: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_set_pitch: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_audio_set_pitch: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let m = unsafe { &mut *mixer };
        match m.set_pitch(MixerHandle(handle), pitch.max(0.01)) {
            Ok(()) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_audio_set_pitch: {}", e));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Pause a currently-playing sound.
///
/// # Safety
///
/// `mixer` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_pause(
    mixer: *mut SoftwareMixer,
    handle: u64,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_pause: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_audio_pause: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let m = unsafe { &mut *mixer };
        match m.pause(MixerHandle(handle)) {
            Ok(()) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_audio_pause: {}", e));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Resume a paused sound.
///
/// # Safety
///
/// `mixer` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_resume(
    mixer: *mut SoftwareMixer,
    handle: u64,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_resume: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_audio_resume: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let m = unsafe { &mut *mixer };
        match m.resume(MixerHandle(handle)) {
            Ok(()) => FFI_OK,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_audio_resume: {}", e));
                FFI_ERR_INVALID_HANDLE
            }
        }
    })
}

/// Query whether a sound is currently playing (not paused, not finished).
///
/// Writes `1` to `out_playing` if playing, `0` otherwise.
///
/// # Safety
///
/// `mixer` and `out_playing` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_is_playing(
    mixer: *mut SoftwareMixer,
    handle: u64,
    out_playing: *mut u8,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_is_playing: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_playing.is_null() {
            set_last_error("genovo_audio_is_playing: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if handle == 0 {
            set_last_error("genovo_audio_is_playing: invalid handle 0");
            return FFI_ERR_INVALID_HANDLE;
        }
        let m = unsafe { &*mixer };
        let playing = match m.channel_state(MixerHandle(handle)) {
            Ok(ChannelState::Playing) | Ok(ChannelState::FadingIn) => true,
            _ => false,
        };
        unsafe {
            *out_playing = if playing { 1 } else { 0 };
        }
        FFI_OK
    })
}

/// Set the master volume of the mixer (affects all playing sounds).
///
/// # Safety
///
/// `mixer` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_set_master_volume(
    mixer: *mut SoftwareMixer,
    volume: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_set_master_volume: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let m = unsafe { &mut *mixer };
        m.set_master_volume(volume.clamp(0.0, 10.0));
        FFI_OK
    })
}

/// Stop all currently playing sounds on the mixer.
///
/// # Safety
///
/// `mixer` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_stop_all(mixer: *mut SoftwareMixer) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_stop_all: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let m = unsafe { &mut *mixer };
        m.stop_all();
        FFI_OK
    })
}

/// Set the 3D position of the audio listener (for spatial audio).
///
/// # Safety
///
/// `mixer` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_set_listener_position(
    mixer: *mut SoftwareMixer,
    x: f32,
    y: f32,
    z: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_set_listener_position: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        // SoftwareMixer does not have a set_listener method.
        // Listener position is managed by AudioSystem at the ECS level.
        // This FFI call is a no-op for now.
        let _m = unsafe { &mut *mixer };
        let _listener = AudioListener {
            position: Vec3::new(x, y, z),
            velocity: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            volume: 1.0,
        };
        FFI_OK
    })
}

/// Set the full 3D listener transform (position, forward, up) for spatial
/// audio.
///
/// # Safety
///
/// `mixer` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_audio_set_listener_orientation(
    mixer: *mut SoftwareMixer,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    fwd_x: f32,
    fwd_y: f32,
    fwd_z: f32,
    up_x: f32,
    up_y: f32,
    up_z: f32,
) -> FfiResult {
    ffi_catch_result(move || {
        if mixer.is_null() {
            set_last_error("genovo_audio_set_listener_orientation: null mixer pointer");
            return FFI_ERR_NULL_POINTER;
        }
        // SoftwareMixer does not have a set_listener method.
        // Listener orientation is managed by AudioSystem at the ECS level.
        // This FFI call is a no-op for now.
        let _m = unsafe { &mut *mixer };
        let _listener = AudioListener {
            position: Vec3::new(pos_x, pos_y, pos_z),
            velocity: Vec3::ZERO,
            forward: Vec3::new(fwd_x, fwd_y, fwd_z),
            up: Vec3::new(up_x, up_y, up_z),
            volume: 1.0,
        };
        FFI_OK
    })
}

use genovo_audio::AudioListener;

// ============================================================================
// Memory management
// ============================================================================

/// Allocate `size` bytes of memory aligned to `align` using the Rust global
/// allocator.
///
/// The returned pointer must be freed with `genovo_free` using the **same**
/// `size` and `align` values.
///
/// Returns null on failure (check `genovo_get_last_error`).
///
/// # Safety
///
/// - `size` must be greater than zero.
/// - `align` must be a power of two and non-zero.
/// - The caller is responsible for eventually freeing the allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_alloc(size: usize, align: usize) -> *mut u8 {
    ffi_catch(ptr::null_mut(), move || {
        if size == 0 {
            set_last_error("genovo_alloc: size must be > 0");
            return ptr::null_mut();
        }
        if align == 0 || !align.is_power_of_two() {
            set_last_error("genovo_alloc: align must be a non-zero power of two");
            return ptr::null_mut();
        }
        let layout = match std::alloc::Layout::from_size_align(size, align) {
            Ok(l) => l,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_alloc: invalid layout: {}", e));
                return ptr::null_mut();
            }
        };
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            set_last_error("genovo_alloc: allocation failed (out of memory)");
        }
        ptr
    })
}

/// Free memory previously allocated with `genovo_alloc`.
///
/// # Safety
///
/// - `ptr` must have been returned by `genovo_alloc` with the **same** `size`
///   and `align` values.
/// - `ptr` must not have been freed already.
/// - After this call, `ptr` is dangling and must not be used.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_free(ptr: *mut u8, size: usize, align: usize) {
    if ptr.is_null() {
        return;
    }
    if size == 0 {
        return;
    }
    if align == 0 || !align.is_power_of_two() {
        set_last_error("genovo_free: align must be a non-zero power of two");
        return;
    }
    if let Ok(layout) = std::alloc::Layout::from_size_align(size, align) {
        unsafe { std::alloc::dealloc(ptr, layout) };
    } else {
        set_last_error("genovo_free: invalid layout parameters");
    }
}

/// Reallocate a block previously returned by `genovo_alloc`.
///
/// The alignment must match the original allocation. If the new size is
/// larger, the extra bytes are zeroed.
///
/// Returns a new pointer (the old one is invalid). Returns null on failure,
/// in which case the original allocation is **not** freed.
///
/// # Safety
///
/// - `ptr` must have been returned by `genovo_alloc` or `genovo_realloc`.
/// - `old_size` and `align` must match the current allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_realloc(
    ptr: *mut u8,
    old_size: usize,
    new_size: usize,
    align: usize,
) -> *mut u8 {
    ffi_catch(ptr::null_mut(), move || {
        if ptr.is_null() {
            // Treat as a fresh allocation.
            return unsafe { genovo_alloc(new_size, align) };
        }
        if new_size == 0 {
            // Treat as free.
            unsafe { genovo_free(ptr, old_size, align) };
            return ptr::null_mut();
        }
        if align == 0 || !align.is_power_of_two() {
            set_last_error("genovo_realloc: align must be a non-zero power of two");
            return ptr::null_mut();
        }
        let old_layout = match std::alloc::Layout::from_size_align(old_size, align) {
            Ok(l) => l,
            Err(e) => {
                set_last_error_fmt(format_args!("genovo_realloc: invalid old layout: {}", e));
                return ptr::null_mut();
            }
        };
        let new_ptr = unsafe { std::alloc::realloc(ptr, old_layout, new_size) };
        if new_ptr.is_null() {
            set_last_error("genovo_realloc: reallocation failed (out of memory)");
            return ptr::null_mut();
        }
        // Zero the extra bytes if the allocation grew.
        if new_size > old_size {
            unsafe {
                ptr::write_bytes(new_ptr.add(old_size), 0, new_size - old_size);
            }
        }
        new_ptr
    })
}

// ============================================================================
// Version / info queries
// ============================================================================

/// Engine version (major).
const VERSION_MAJOR: u32 = 0;
/// Engine version (minor).
const VERSION_MINOR: u32 = 1;
/// Engine version (patch).
const VERSION_PATCH: u32 = 0;

/// Packed version number: (major << 22) | (minor << 12) | patch.
const VERSION_PACKED: u32 = (VERSION_MAJOR << 22) | (VERSION_MINOR << 12) | VERSION_PATCH;

/// Return the packed version number of the FFI layer.
///
/// Bits 31..22 = major, 21..12 = minor, 11..0 = patch.
#[unsafe(no_mangle)]
pub extern "C" fn genovo_version() -> u32 {
    VERSION_PACKED
}

/// Return the major version component.
#[unsafe(no_mangle)]
pub extern "C" fn genovo_version_major() -> u32 {
    VERSION_MAJOR
}

/// Return the minor version component.
#[unsafe(no_mangle)]
pub extern "C" fn genovo_version_minor() -> u32 {
    VERSION_MINOR
}

/// Return the patch version component.
#[unsafe(no_mangle)]
pub extern "C" fn genovo_version_patch() -> u32 {
    VERSION_PATCH
}

/// Return a static C string describing the engine version.
///
/// The returned pointer is valid for the lifetime of the process.
#[unsafe(no_mangle)]
pub extern "C" fn genovo_version_string() -> *const c_char {
    // Using a byte literal with null terminator avoids dynamic allocation.
    static VERSION_STR: &[u8] = b"Genovo 0.1.0\0";
    VERSION_STR.as_ptr() as *const c_char
}

// ============================================================================
// Utility: string helpers for C consumers
// ============================================================================

/// Free a C string that was allocated by the FFI layer.
///
/// Some FFI functions return dynamically-allocated C strings. This function
/// frees them.
///
/// # Safety
///
/// `ptr` must be a C string allocated by this FFI layer, or null (in which
/// case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

// ============================================================================
// Initialization / shutdown
// ============================================================================

/// Global initialization of the Genovo engine.
///
/// Must be called exactly once before using any other FFI function (except
/// version queries, which are always safe). Initializes logging, global
/// allocator hooks, and other subsystem prerequisites.
///
/// Returns `FFI_OK` on success, or a negative error code on failure.
#[unsafe(no_mangle)]
pub extern "C" fn genovo_init() -> FfiResult {
    ffi_catch_result(|| {
        // Initialize logging (best-effort; do not fail if it has already been
        // set up by a previous call).
        #[cfg(feature = "log-init")]
        {
            let _ = env_logger::try_init();
        }
        log::info!("Genovo FFI initialized (version {})", VERSION_PACKED);
        FFI_OK
    })
}

/// Global shutdown of the Genovo engine.
///
/// Should be called when the host application is done using the engine.
/// After this call, no other FFI functions should be invoked except
/// `genovo_init`.
///
/// # Safety
///
/// All engine objects (worlds, mixers, clips, etc.) must have been destroyed
/// before calling this function.
#[unsafe(no_mangle)]
pub extern "C" fn genovo_shutdown() -> FfiResult {
    ffi_catch_result(|| {
        log::info!("Genovo FFI shutting down");
        FFI_OK
    })
}

// ============================================================================
// ECS FFI (minimal surface for C consumers)
// ============================================================================

use genovo_ecs::World as EcsWorld;

/// Create a new ECS world.
///
/// Returns a heap-allocated `EcsWorld` as an opaque pointer. The caller must
/// eventually free it with `genovo_ecs_world_destroy`.
///
/// # Safety
///
/// Caller owns the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_ecs_world_create() -> *mut EcsWorld {
    ffi_catch(ptr::null_mut(), || {
        let world = EcsWorld::new();
        Box::into_raw(Box::new(world))
    })
}

/// Destroy an ECS world previously created with `genovo_ecs_world_create`.
///
/// # Safety
///
/// `world` must be a valid pointer and must not have been previously destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_ecs_world_destroy(world: *mut EcsWorld) -> FfiResult {
    ffi_catch_result(|| {
        if world.is_null() {
            set_last_error("genovo_ecs_world_destroy: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        unsafe {
            let _ = Box::from_raw(world);
        }
        FFI_OK
    })
}

/// Create an entity in the ECS world.
///
/// Returns the entity ID as a `u64`. Returns `0` on failure.
///
/// # Safety
///
/// `world` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_ecs_create_entity(world: *mut EcsWorld) -> u64 {
    ffi_catch(0u64, move || {
        if world.is_null() {
            set_last_error("genovo_ecs_create_entity: null world pointer");
            return 0;
        }
        let w = unsafe { &mut *world };
        let entity = w.spawn_entity().build();
        entity.id as u64
    })
}

/// Destroy an entity in the ECS world.
///
/// # Safety
///
/// `world` must be valid and non-null. `entity_id` must be a valid entity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_ecs_destroy_entity(
    world: *mut EcsWorld,
    entity_id: u64,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_ecs_destroy_entity: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let w = unsafe { &mut *world };
        let entity = genovo_ecs::Entity::new(entity_id as u32, 0);
        w.despawn(entity);
        FFI_OK
    })
}

/// Query the number of live entities in the ECS world.
///
/// # Safety
///
/// `world` and `out_count` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn genovo_ecs_entity_count(
    world: *mut EcsWorld,
    out_count: *mut u32,
) -> FfiResult {
    ffi_catch_result(move || {
        if world.is_null() {
            set_last_error("genovo_ecs_entity_count: null world pointer");
            return FFI_ERR_NULL_POINTER;
        }
        if out_count.is_null() {
            set_last_error("genovo_ecs_entity_count: null output pointer");
            return FFI_ERR_NULL_POINTER;
        }
        let w = unsafe { &*world };
        unsafe {
            *out_count = w.entity_count() as u32;
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

    // ---- FfiVec3 tests ----

    #[test]
    fn test_ffi_vec3_default() {
        let v = FfiVec3::default();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn test_ffi_vec3_new() {
        let v = FfiVec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_ffi_vec3_roundtrip() {
        let original = Vec3::new(1.5, -2.5, 3.5);
        let ffi: FfiVec3 = original.into();
        let back: Vec3 = ffi.into();
        assert_eq!(back.x, original.x);
        assert_eq!(back.y, original.y);
        assert_eq!(back.z, original.z);
    }

    // ---- FfiQuat tests ----

    #[test]
    fn test_ffi_quat_default_is_identity() {
        let q = FfiQuat::default();
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
        assert_eq!(q.w, 1.0);
    }

    #[test]
    fn test_ffi_quat_roundtrip() {
        let original = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9);
        let ffi: FfiQuat = original.into();
        let back: Quat = ffi.into();
        assert_eq!(back.x, original.x);
        assert_eq!(back.y, original.y);
        assert_eq!(back.z, original.z);
        assert_eq!(back.w, original.w);
    }

    // ---- FfiTransform tests ----

    #[test]
    fn test_ffi_transform_default_is_identity() {
        let t = FfiTransform::default();
        assert_eq!(t.position, FfiVec3::ZERO);
        assert_eq!(t.rotation, FfiQuat::IDENTITY);
        assert_eq!(t.scale, FfiVec3::ONE);
    }

    // ---- FfiRigidBodyDesc tests ----

    #[test]
    fn test_ffi_rigid_body_desc_default() {
        let desc = FfiRigidBodyDesc::default();
        assert_eq!(desc.body_type, 1); // Dynamic
        assert_eq!(desc.mass, 1.0);
    }

    #[test]
    fn test_body_type_from_u8_valid() {
        assert_eq!(body_type_from_u8(0), Some(BodyType::Static));
        assert_eq!(body_type_from_u8(1), Some(BodyType::Dynamic));
        assert_eq!(body_type_from_u8(2), Some(BodyType::Kinematic));
    }

    #[test]
    fn test_body_type_from_u8_invalid() {
        assert_eq!(body_type_from_u8(3), None);
        assert_eq!(body_type_from_u8(255), None);
    }

    // ---- Error handling tests ----

    #[test]
    fn test_error_set_and_clear() {
        set_last_error("test error message");
        let ptr = genovo_get_last_error();
        assert!(!ptr.is_null());
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "test error message");

        genovo_clear_error();
        let ptr2 = genovo_get_last_error();
        assert!(ptr2.is_null());
    }

    // ---- Memory FFI tests ----

    #[test]
    fn test_alloc_and_free() {
        unsafe {
            let ptr = genovo_alloc(256, 8);
            assert!(!ptr.is_null());
            // Write a pattern to verify the memory is usable.
            for i in 0..256 {
                *ptr.add(i) = (i & 0xFF) as u8;
            }
            genovo_free(ptr, 256, 8);
        }
    }

    #[test]
    fn test_alloc_zero_size_returns_null() {
        unsafe {
            let ptr = genovo_alloc(0, 8);
            assert!(ptr.is_null());
        }
    }

    #[test]
    fn test_alloc_bad_align_returns_null() {
        unsafe {
            let ptr = genovo_alloc(64, 3); // 3 is not a power of two
            assert!(ptr.is_null());
        }
    }

    #[test]
    fn test_free_null_is_noop() {
        unsafe {
            genovo_free(ptr::null_mut(), 64, 8); // Should not crash.
        }
    }

    #[test]
    fn test_realloc_grow() {
        unsafe {
            let ptr = genovo_alloc(64, 8);
            assert!(!ptr.is_null());
            *ptr = 42;
            let ptr2 = genovo_realloc(ptr, 64, 128, 8);
            assert!(!ptr2.is_null());
            assert_eq!(*ptr2, 42); // Original data preserved.
            genovo_free(ptr2, 128, 8);
        }
    }

    #[test]
    fn test_realloc_null_acts_as_alloc() {
        unsafe {
            let ptr = genovo_realloc(ptr::null_mut(), 0, 64, 8);
            assert!(!ptr.is_null());
            genovo_free(ptr, 64, 8);
        }
    }

    // ---- Version tests ----

    #[test]
    fn test_version_components() {
        assert_eq!(genovo_version_major(), 0);
        assert_eq!(genovo_version_minor(), 1);
        assert_eq!(genovo_version_patch(), 0);
    }

    #[test]
    fn test_version_packed() {
        let packed = genovo_version();
        let major = packed >> 22;
        let minor = (packed >> 12) & 0x3FF;
        let patch = packed & 0xFFF;
        assert_eq!(major, VERSION_MAJOR);
        assert_eq!(minor, VERSION_MINOR);
        assert_eq!(patch, VERSION_PATCH);
    }

    #[test]
    fn test_version_string_not_null() {
        let s = genovo_version_string();
        assert!(!s.is_null());
        let cstr = unsafe { std::ffi::CStr::from_ptr(s) };
        assert!(cstr.to_str().unwrap().contains("Genovo"));
    }

    // ---- Init / shutdown ----

    #[test]
    fn test_init_shutdown() {
        assert_eq!(genovo_init(), FFI_OK);
        assert_eq!(genovo_shutdown(), FFI_OK);
    }

    // ---- Physics null-pointer checks ----

    #[test]
    fn test_physics_world_destroy_null() {
        unsafe {
            let result = genovo_physics_world_destroy(ptr::null_mut());
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    #[test]
    fn test_physics_world_step_null() {
        unsafe {
            let result = genovo_physics_world_step(ptr::null_mut(), 0.016);
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    #[test]
    fn test_physics_add_body_null_world() {
        unsafe {
            let desc = FfiRigidBodyDesc::default();
            let handle = genovo_physics_add_body(ptr::null_mut(), &desc);
            assert_eq!(handle, 0);
        }
    }

    #[test]
    fn test_physics_add_body_null_desc() {
        unsafe {
            // We need a real world for this test.
            let world = genovo_physics_world_create(0.0, -9.81, 0.0);
            assert!(!world.is_null());
            let handle = genovo_physics_add_body(world, ptr::null());
            assert_eq!(handle, 0);
            genovo_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_physics_remove_body_null() {
        unsafe {
            let result = genovo_physics_remove_body(ptr::null_mut(), 1);
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    #[test]
    fn test_physics_set_gravity_null() {
        unsafe {
            let result = genovo_physics_world_set_gravity(ptr::null_mut(), 0.0, -9.81, 0.0);
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    // ---- Audio null-pointer checks ----

    #[test]
    fn test_audio_mixer_destroy_null() {
        unsafe {
            let result = genovo_audio_mixer_destroy(ptr::null_mut());
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    #[test]
    fn test_audio_mixer_update_null() {
        unsafe {
            let result = genovo_audio_mixer_update(ptr::null_mut(), 0.016);
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    #[test]
    fn test_audio_play_null_mixer() {
        unsafe {
            let handle = genovo_audio_play(ptr::null_mut(), ptr::null(), 1.0, 1.0, 0);
            assert_eq!(handle, 0);
        }
    }

    #[test]
    fn test_audio_stop_null() {
        unsafe {
            let result = genovo_audio_stop(ptr::null_mut(), 1);
            assert_eq!(result, FFI_ERR_NULL_POINTER);
        }
    }

    #[test]
    fn test_audio_clip_from_wav_null() {
        unsafe {
            let clip = genovo_audio_clip_from_wav(ptr::null(), 0);
            assert!(clip.is_null());
        }
    }

    // ---- Raycast null-pointer checks ----

    #[test]
    fn test_raycast_null_world() {
        unsafe {
            let mut result = FfiRaycastResult::default();
            let count = genovo_physics_raycast(
                ptr::null_mut(),
                0.0, 0.0, 0.0,
                0.0, -1.0, 0.0,
                100.0,
                &mut result,
                1,
            );
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn test_raycast_null_output() {
        unsafe {
            let world = genovo_physics_world_create(0.0, -9.81, 0.0);
            let count = genovo_physics_raycast(
                world,
                0.0, 0.0, 0.0,
                0.0, -1.0, 0.0,
                100.0,
                ptr::null_mut(),
                1,
            );
            assert_eq!(count, 0);
            genovo_physics_world_destroy(world);
        }
    }

    // ---- catch_unwind test ----

    #[test]
    fn test_ffi_catch_handles_panic() {
        let result = ffi_catch_result(|| {
            panic!("intentional test panic");
        });
        assert_eq!(result, FFI_ERR_PANIC);
        let err = genovo_get_last_error();
        assert!(!err.is_null());
    }
}
