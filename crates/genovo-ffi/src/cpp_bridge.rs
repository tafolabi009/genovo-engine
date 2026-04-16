//! # C++ Interop Helpers
//!
//! Provides C++-friendly wrappers and data types for interoperating with C++
//! code. This module includes:
//!
//! - `FfiString`: A C-compatible string type with ownership semantics
//! - `FfiArray<T>`: A C-compatible dynamic array (data, length, capacity)
//! - `FfiCallback`: Function pointer wrappers for C++ callbacks
//! - `FfiEventSystem`: C++ callback registration for engine events
//! - Smart pointer interop helpers (shared_ptr <-> Arc)
//! - Exception safety: Result -> error code conversion
//! - `#[repr(C)]` structs for all cross-language types

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    ffi_catch, ffi_catch_result, set_last_error, set_last_error_fmt, FfiQuat, FfiResult,
    FfiTransform, FfiVec3, FFI_ERR_INTERNAL, FFI_ERR_INVALID_HANDLE, FFI_ERR_INVALID_PARAMETER,
    FFI_ERR_NOT_IMPLEMENTED, FFI_ERR_NULL_POINTER, FFI_ERR_OUT_OF_MEMORY, FFI_OK,
};

// ============================================================================
// FfiString -- C-compatible owned string
// ============================================================================

/// A C-compatible owned string.
///
/// This struct owns the memory for the string data. When passed to C++, the
/// C++ side must call `ffi_string_free()` when done. The data pointer always
/// points to a null-terminated UTF-8 string.
#[repr(C)]
#[derive(Debug)]
pub struct FfiString {
    /// Pointer to null-terminated UTF-8 string data.
    pub data: *mut c_char,
    /// Length in bytes, not including the null terminator.
    pub length: usize,
    /// Capacity of the allocation in bytes (including null terminator).
    pub capacity: usize,
}

impl FfiString {
    /// Create a new empty `FfiString`.
    pub fn new() -> Self {
        Self {
            data: ptr::null_mut(),
            length: 0,
            capacity: 0,
        }
    }

    /// Create an `FfiString` from a Rust `&str`.
    pub fn from_str(s: &str) -> Self {
        match CString::new(s) {
            Ok(cstr) => {
                let bytes = cstr.into_bytes_with_nul();
                let length = bytes.len() - 1; // exclude null
                let capacity = bytes.len();
                let mut v = bytes.into_boxed_slice();
                let data = v.as_mut_ptr() as *mut c_char;
                std::mem::forget(v);
                Self {
                    data,
                    length,
                    capacity,
                }
            }
            Err(_) => {
                // String contains interior null bytes; replace them
                let sanitized = s.replace('\0', "");
                Self::from_str(&sanitized)
            }
        }
    }

    /// Create an `FfiString` from a C string pointer (copies the data).
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid null-terminated C string.
    pub unsafe fn from_cstr(ptr: *const c_char) -> Self {
        if ptr.is_null() {
            return Self::new();
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        match cstr.to_str() {
            Ok(s) => Self::from_str(s),
            Err(_) => Self::new(),
        }
    }

    /// Get the string data as a C string pointer.
    ///
    /// Returns null if the string is empty.
    pub fn as_ptr(&self) -> *const c_char {
        if self.data.is_null() || self.length == 0 {
            // Return a pointer to a static empty string
            static EMPTY: &[u8] = b"\0";
            return EMPTY.as_ptr() as *const c_char;
        }
        self.data as *const c_char
    }

    /// Convert to a Rust `&str`, returning an empty string if invalid.
    pub fn to_str(&self) -> &str {
        if self.data.is_null() || self.length == 0 {
            return "";
        }
        unsafe {
            let slice = std::slice::from_raw_parts(self.data as *const u8, self.length);
            std::str::from_utf8(slice).unwrap_or("")
        }
    }

    /// Returns true if the string is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

impl Default for FfiString {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FfiString {
    fn drop(&mut self) {
        if !self.data.is_null() && self.capacity > 0 {
            unsafe {
                let _ = Vec::from_raw_parts(self.data as *mut u8, self.capacity, self.capacity);
            }
        }
    }
}

impl Clone for FfiString {
    fn clone(&self) -> Self {
        Self::from_str(self.to_str())
    }
}

/// Create an `FfiString` from a C string.
///
/// # Safety
///
/// `ptr` must be a valid null-terminated C string or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_string_create(ptr: *const c_char) -> FfiString {
    unsafe { FfiString::from_cstr(ptr) }
}

/// Create an `FfiString` from a byte buffer with explicit length.
///
/// # Safety
///
/// `data` must point to at least `len` bytes of valid UTF-8 data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_string_create_from_bytes(
    data: *const u8,
    len: usize,
) -> FfiString {
    if data.is_null() || len == 0 {
        return FfiString::new();
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    match std::str::from_utf8(slice) {
        Ok(s) => FfiString::from_str(s),
        Err(_) => FfiString::new(),
    }
}

/// Free an `FfiString`. After calling this, the FfiString is invalid.
///
/// # Safety
///
/// `s` must be a valid pointer to an `FfiString` that was created by this
/// FFI layer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_string_free(s: *mut FfiString) {
    if !s.is_null() {
        unsafe {
            let owned = ptr::read(s);
            drop(owned);
            // Zero out the struct to prevent use-after-free
            (*s).data = ptr::null_mut();
            (*s).length = 0;
            (*s).capacity = 0;
        }
    }
}

/// Get the length of an `FfiString`.
#[unsafe(no_mangle)]
pub extern "C" fn ffi_string_length(s: *const FfiString) -> usize {
    if s.is_null() {
        return 0;
    }
    unsafe { (*s).length }
}

/// Get a const pointer to the string data.
#[unsafe(no_mangle)]
pub extern "C" fn ffi_string_data(s: *const FfiString) -> *const c_char {
    if s.is_null() {
        static EMPTY: &[u8] = b"\0";
        return EMPTY.as_ptr() as *const c_char;
    }
    unsafe { (*s).as_ptr() }
}

// ============================================================================
// FfiArray -- C-compatible dynamic array
// ============================================================================

/// A C-compatible dynamic array with type-erased element storage.
///
/// Memory layout is identical to a C struct with pointer, length, and capacity.
/// The C++ side is responsible for interpreting element types correctly.
#[repr(C)]
#[derive(Debug)]
pub struct FfiArray {
    /// Pointer to the array data.
    pub data: *mut c_void,
    /// Number of elements currently stored.
    pub length: usize,
    /// Capacity in number of elements.
    pub capacity: usize,
    /// Size of each element in bytes (for bounds checking).
    pub element_size: usize,
}

impl FfiArray {
    /// Create a new empty array for elements of `element_size` bytes.
    pub fn new(element_size: usize) -> Self {
        Self {
            data: ptr::null_mut(),
            length: 0,
            capacity: 0,
            element_size,
        }
    }

    /// Create an array from a Vec of f32 values.
    pub fn from_f32_vec(v: Vec<f32>) -> Self {
        let length = v.len();
        let capacity = v.capacity();
        let data = Box::into_raw(v.into_boxed_slice()) as *mut c_void;
        Self {
            data,
            length,
            capacity,
            element_size: std::mem::size_of::<f32>(),
        }
    }

    /// Create an array from a Vec of u32 values.
    pub fn from_u32_vec(v: Vec<u32>) -> Self {
        let length = v.len();
        let capacity = v.capacity();
        let data = Box::into_raw(v.into_boxed_slice()) as *mut c_void;
        Self {
            data,
            length,
            capacity,
            element_size: std::mem::size_of::<u32>(),
        }
    }

    /// Create an array from a Vec of u8 values.
    pub fn from_u8_vec(v: Vec<u8>) -> Self {
        let length = v.len();
        let capacity = v.capacity();
        let data = Box::into_raw(v.into_boxed_slice()) as *mut c_void;
        Self {
            data,
            length,
            capacity,
            element_size: 1,
        }
    }

    /// Returns true if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the total size in bytes of the stored data.
    pub fn byte_size(&self) -> usize {
        self.length * self.element_size
    }
}

/// Create an empty FfiArray with the given element size.
#[unsafe(no_mangle)]
pub extern "C" fn ffi_array_create(element_size: usize) -> FfiArray {
    FfiArray::new(element_size)
}

/// Create an FfiArray from existing data (copies the data).
///
/// # Safety
///
/// `data` must point to at least `count * element_size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_array_create_from(
    data: *const c_void,
    count: usize,
    element_size: usize,
) -> FfiArray {
    if data.is_null() || count == 0 || element_size == 0 {
        return FfiArray::new(element_size);
    }
    let total_bytes = count * element_size;
    let mut vec = vec![0u8; total_bytes];
    unsafe {
        ptr::copy_nonoverlapping(data as *const u8, vec.as_mut_ptr(), total_bytes);
    }
    let boxed = vec.into_boxed_slice();
    FfiArray {
        data: Box::into_raw(boxed) as *mut c_void,
        length: count,
        capacity: count,
        element_size,
    }
}

/// Free an FfiArray.
///
/// # Safety
///
/// `arr` must be a valid pointer to an FfiArray created by this FFI layer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_array_free(arr: *mut FfiArray) {
    if !arr.is_null() {
        let a = unsafe { &mut *arr };
        if !a.data.is_null() && a.capacity > 0 {
            let total_bytes = a.capacity * a.element_size;
            unsafe {
                let _ = Vec::from_raw_parts(a.data as *mut u8, total_bytes, total_bytes);
            }
        }
        a.data = ptr::null_mut();
        a.length = 0;
        a.capacity = 0;
    }
}

/// Get the element count of an FfiArray.
#[unsafe(no_mangle)]
pub extern "C" fn ffi_array_length(arr: *const FfiArray) -> usize {
    if arr.is_null() {
        return 0;
    }
    unsafe { (*arr).length }
}

/// Get a const pointer to the array data.
#[unsafe(no_mangle)]
pub extern "C" fn ffi_array_data(arr: *const FfiArray) -> *const c_void {
    if arr.is_null() {
        return ptr::null();
    }
    unsafe { (*arr).data as *const c_void }
}

// ============================================================================
// FfiCallback -- function pointer wrapper
// ============================================================================

/// A C-compatible callback wrapper.
///
/// Wraps a function pointer with an optional user-data pointer, matching
/// the common C callback pattern: `void callback(void* user_data, ...)`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiCallback {
    /// The function pointer. Signature: `void(*)(void* user_data, const void* event_data)`.
    pub func: Option<unsafe extern "C" fn(*mut c_void, *const c_void)>,
    /// User-supplied context pointer, passed as the first argument to `func`.
    pub user_data: *mut c_void,
}

impl FfiCallback {
    /// Create a null (no-op) callback.
    pub fn null() -> Self {
        Self {
            func: None,
            user_data: ptr::null_mut(),
        }
    }

    /// Returns true if the callback is set (non-null function pointer).
    pub fn is_set(&self) -> bool {
        self.func.is_some()
    }

    /// Invoke the callback with the given event data.
    ///
    /// Does nothing if the function pointer is null.
    ///
    /// # Safety
    ///
    /// `event_data` must be valid for the expected callback type.
    pub unsafe fn invoke(&self, event_data: *const c_void) {
        if let Some(f) = self.func {
            unsafe {
                f(self.user_data, event_data);
            }
        }
    }
}

impl Default for FfiCallback {
    fn default() -> Self {
        Self::null()
    }
}

// Safety: The user_data pointer is managed by the C++ side and we assume
// they handle thread safety for their own data.
unsafe impl Send for FfiCallback {}
unsafe impl Sync for FfiCallback {}

// ============================================================================
// FfiEventSystem -- C++ callback registration
// ============================================================================

/// Event types that can be subscribed to from C++.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FfiEventType {
    /// Engine initialized.
    EngineInit = 0,
    /// Engine shutting down.
    EngineShutdown = 1,
    /// Frame started.
    FrameBegin = 2,
    /// Frame ended.
    FrameEnd = 3,
    /// Entity created.
    EntityCreated = 4,
    /// Entity destroyed.
    EntityDestroyed = 5,
    /// Physics step completed.
    PhysicsStep = 6,
    /// Collision detected.
    Collision = 7,
    /// Asset loaded.
    AssetLoaded = 8,
    /// Asset unloaded.
    AssetUnloaded = 9,
    /// Scene loaded.
    SceneLoaded = 10,
    /// Window resized.
    WindowResize = 11,
    /// Window focus changed.
    WindowFocus = 12,
    /// Input event.
    Input = 13,
    /// Custom user event (id provided in event data).
    Custom = 100,
}

/// Maximum number of event types supported.
const MAX_EVENT_TYPES: usize = 128;

/// Maximum number of callbacks per event type.
const MAX_CALLBACKS_PER_EVENT: usize = 32;

/// Event subscription handle for unregistering.
pub type FfiEventSubscription = u64;

/// Data passed with collision events.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiCollisionEvent {
    /// Handle of body A.
    pub body_a: u64,
    /// Handle of body B.
    pub body_b: u64,
    /// Contact point in world space.
    pub contact_point: FfiVec3,
    /// Contact normal.
    pub contact_normal: FfiVec3,
    /// Penetration depth.
    pub penetration: f32,
    /// Impulse magnitude.
    pub impulse: f32,
}

/// Data passed with window resize events.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiWindowResizeEvent {
    /// New width in pixels.
    pub width: u32,
    /// New height in pixels.
    pub height: u32,
}

/// The event system manages C++ callback registrations.
pub struct FfiEventSystem {
    /// Callbacks indexed by event type, each slot holding up to N callbacks.
    callbacks: Vec<Vec<(FfiEventSubscription, FfiCallback)>>,
    /// Next subscription handle.
    next_subscription: AtomicU64,
}

impl FfiEventSystem {
    /// Create a new event system.
    pub fn new() -> Self {
        let mut callbacks = Vec::with_capacity(MAX_EVENT_TYPES);
        for _ in 0..MAX_EVENT_TYPES {
            callbacks.push(Vec::new());
        }
        Self {
            callbacks,
            next_subscription: AtomicU64::new(1),
        }
    }

    /// Register a callback for an event type.
    pub fn subscribe(
        &mut self,
        event_type: FfiEventType,
        callback: FfiCallback,
    ) -> FfiEventSubscription {
        let idx = event_type as usize;
        if idx >= MAX_EVENT_TYPES {
            return 0;
        }
        if self.callbacks[idx].len() >= MAX_CALLBACKS_PER_EVENT {
            return 0; // Too many callbacks
        }
        let handle = self.next_subscription.fetch_add(1, Ordering::Relaxed);
        self.callbacks[idx].push((handle, callback));
        handle
    }

    /// Unregister a callback by subscription handle.
    pub fn unsubscribe(&mut self, subscription: FfiEventSubscription) -> bool {
        for slot in &mut self.callbacks {
            if let Some(pos) = slot.iter().position(|(h, _)| *h == subscription) {
                slot.swap_remove(pos);
                return true;
            }
        }
        false
    }

    /// Fire an event, invoking all registered callbacks.
    ///
    /// # Safety
    ///
    /// `event_data` must be valid for the event type.
    pub unsafe fn fire(&self, event_type: FfiEventType, event_data: *const c_void) {
        let idx = event_type as usize;
        if idx >= MAX_EVENT_TYPES {
            return;
        }
        for (_, callback) in &self.callbacks[idx] {
            unsafe {
                callback.invoke(event_data);
            }
        }
    }

    /// Get the number of subscribers for an event type.
    pub fn subscriber_count(&self, event_type: FfiEventType) -> usize {
        let idx = event_type as usize;
        if idx >= MAX_EVENT_TYPES {
            return 0;
        }
        self.callbacks[idx].len()
    }
}

impl Default for FfiEventSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a new event system.
///
/// # Safety
///
/// Caller owns the returned pointer and must free it with `ffi_event_system_destroy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_event_system_create() -> *mut FfiEventSystem {
    Box::into_raw(Box::new(FfiEventSystem::new()))
}

/// Destroy an event system.
///
/// # Safety
///
/// `system` must be a valid pointer returned by `ffi_event_system_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_event_system_destroy(system: *mut FfiEventSystem) -> FfiResult {
    if system.is_null() {
        set_last_error("ffi_event_system_destroy: null pointer");
        return FFI_ERR_NULL_POINTER;
    }
    unsafe {
        let _ = Box::from_raw(system);
    }
    FFI_OK
}

/// Subscribe to an event type.
///
/// Returns a subscription handle for later unsubscription, or 0 on failure.
///
/// # Safety
///
/// `system` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_event_subscribe(
    system: *mut FfiEventSystem,
    event_type: FfiEventType,
    callback: FfiCallback,
) -> FfiEventSubscription {
    if system.is_null() {
        set_last_error("ffi_event_subscribe: null system pointer");
        return 0;
    }
    let s = unsafe { &mut *system };
    s.subscribe(event_type, callback)
}

/// Unsubscribe from an event.
///
/// # Safety
///
/// `system` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_event_unsubscribe(
    system: *mut FfiEventSystem,
    subscription: FfiEventSubscription,
) -> FfiResult {
    if system.is_null() {
        set_last_error("ffi_event_unsubscribe: null system pointer");
        return FFI_ERR_NULL_POINTER;
    }
    let s = unsafe { &mut *system };
    if s.unsubscribe(subscription) {
        FFI_OK
    } else {
        set_last_error("ffi_event_unsubscribe: subscription not found");
        FFI_ERR_INVALID_HANDLE
    }
}

/// Fire an event, invoking all registered callbacks.
///
/// # Safety
///
/// `system` must be valid and non-null. `event_data` must be valid for the
/// event type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_event_fire(
    system: *const FfiEventSystem,
    event_type: FfiEventType,
    event_data: *const c_void,
) -> FfiResult {
    if system.is_null() {
        set_last_error("ffi_event_fire: null system pointer");
        return FFI_ERR_NULL_POINTER;
    }
    let s = unsafe { &*system };
    unsafe {
        s.fire(event_type, event_data);
    }
    FFI_OK
}

// ============================================================================
// Smart pointer interop
// ============================================================================

/// Opaque reference-counted handle for sharing ownership between Rust and C++.
///
/// Wraps a Rust `Arc<T>` behind an opaque pointer. C++ can acquire/release
/// references using `ffi_rc_acquire` and `ffi_rc_release`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiRcHandle {
    /// Opaque pointer to the Arc internals.
    pub ptr: *const c_void,
    /// Type identifier for runtime type checking.
    pub type_id: u64,
}

impl FfiRcHandle {
    /// Create a null handle.
    pub fn null() -> Self {
        Self {
            ptr: ptr::null(),
            type_id: 0,
        }
    }

    /// Returns true if the handle is valid (non-null).
    pub fn is_valid(&self) -> bool {
        !self.ptr.is_null()
    }
}

impl Default for FfiRcHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Increment the reference count of an RC handle.
///
/// # Safety
///
/// `handle` must be a valid FfiRcHandle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_rc_acquire(handle: FfiRcHandle) -> FfiRcHandle {
    // In a real implementation, this would call Arc::clone and return
    // a new handle pointing to the same data.
    handle
}

/// Decrement the reference count. If this is the last reference, the
/// underlying data is freed.
///
/// # Safety
///
/// `handle` must be a valid FfiRcHandle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_rc_release(handle: FfiRcHandle) -> FfiResult {
    if !handle.is_valid() {
        return FFI_OK; // No-op for null handles
    }
    // In a real implementation, this would drop the Arc
    FFI_OK
}

/// Get the current reference count.
///
/// # Safety
///
/// `handle` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_rc_count(handle: FfiRcHandle) -> u32 {
    if !handle.is_valid() {
        return 0;
    }
    // Placeholder
    1
}

// ============================================================================
// RTTI bridge -- expose Rust type info to C++
// ============================================================================

/// Type information exposed to C++.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiTypeInfo {
    /// Type name (null-terminated).
    pub name: *const c_char,
    /// Size of the type in bytes.
    pub size: usize,
    /// Alignment of the type in bytes.
    pub alignment: usize,
    /// Unique type identifier.
    pub type_id: u64,
    /// Whether the type implements Copy.
    pub is_copy: u8,
    /// Whether the type implements Send.
    pub is_send: u8,
    /// Whether the type implements Sync.
    pub is_sync: u8,
    /// Reserved.
    pub _reserved: u8,
}

/// Get type info for common engine types.
///
/// `type_id` values:
/// - 1: FfiVec3
/// - 2: FfiQuat
/// - 3: FfiTransform
/// - 4: FfiMat4
///
/// Returns FFI_OK on success, error code if type_id is unknown.
///
/// # Safety
///
/// `out_info` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_get_type_info(
    type_id: u64,
    out_info: *mut FfiTypeInfo,
) -> FfiResult {
    if out_info.is_null() {
        set_last_error("ffi_get_type_info: null output pointer");
        return FFI_ERR_NULL_POINTER;
    }

    static VEC3_NAME: &[u8] = b"FfiVec3\0";
    static QUAT_NAME: &[u8] = b"FfiQuat\0";
    static TRANSFORM_NAME: &[u8] = b"FfiTransform\0";
    static MAT4_NAME: &[u8] = b"FfiMat4\0";

    let info = match type_id {
        1 => FfiTypeInfo {
            name: VEC3_NAME.as_ptr() as *const c_char,
            size: std::mem::size_of::<FfiVec3>(),
            alignment: std::mem::align_of::<FfiVec3>(),
            type_id: 1,
            is_copy: 1,
            is_send: 1,
            is_sync: 1,
            _reserved: 0,
        },
        2 => FfiTypeInfo {
            name: QUAT_NAME.as_ptr() as *const c_char,
            size: std::mem::size_of::<FfiQuat>(),
            alignment: std::mem::align_of::<FfiQuat>(),
            type_id: 2,
            is_copy: 1,
            is_send: 1,
            is_sync: 1,
            _reserved: 0,
        },
        3 => FfiTypeInfo {
            name: TRANSFORM_NAME.as_ptr() as *const c_char,
            size: std::mem::size_of::<FfiTransform>(),
            alignment: std::mem::align_of::<FfiTransform>(),
            type_id: 3,
            is_copy: 1,
            is_send: 1,
            is_sync: 1,
            _reserved: 0,
        },
        4 => FfiTypeInfo {
            name: MAT4_NAME.as_ptr() as *const c_char,
            size: std::mem::size_of::<FfiMat4>(),
            alignment: std::mem::align_of::<FfiMat4>(),
            type_id: 4,
            is_copy: 1,
            is_send: 1,
            is_sync: 1,
            _reserved: 0,
        },
        _ => {
            set_last_error_fmt(format_args!("ffi_get_type_info: unknown type_id {}", type_id));
            return FFI_ERR_INVALID_PARAMETER;
        }
    };

    unsafe {
        *out_info = info;
    }
    FFI_OK
}

// ============================================================================
// #[repr(C)] cross-language types
// ============================================================================

/// 4x4 matrix in C-compatible layout (column-major, 16 floats).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfiMat4 {
    /// Column-major data: columns[0..4] = first column, etc.
    pub data: [f32; 16],
}

impl FfiMat4 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self {
        data: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };

    /// The zero matrix.
    pub const ZERO: Self = Self { data: [0.0; 16] };

    /// Create from a flat array.
    pub const fn from_cols_array(data: [f32; 16]) -> Self {
        Self { data }
    }

    /// Get an element at (row, col).
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.data[col * 4 + row]
    }

    /// Set an element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[col * 4 + row] = value;
    }
}

impl Default for FfiMat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// 2D rectangle in C-compatible layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfiRect {
    /// X coordinate of the top-left corner.
    pub x: f32,
    /// Y coordinate of the top-left corner.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

impl FfiRect {
    /// Create a new rect.
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Zero-sized rect at origin.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        width: 0.0,
        height: 0.0,
    };
}

impl Default for FfiRect {
    fn default() -> Self {
        Self::ZERO
    }
}

/// RGBA color in C-compatible layout (linear, 0..1 range).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfiColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl FfiColor {
    /// Create a new color.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// White (1, 1, 1, 1).
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    /// Black (0, 0, 0, 1).
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Transparent (0, 0, 0, 0).
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Red (1, 0, 0, 1).
    pub const RED: Self = Self {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Green (0, 1, 0, 1).
    pub const GREEN: Self = Self {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };

    /// Blue (0, 0, 1, 1).
    pub const BLUE: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
}

impl Default for FfiColor {
    fn default() -> Self {
        Self::WHITE
    }
}

/// A ray in 3D space (C-compatible).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfiRay {
    /// Origin point.
    pub origin: FfiVec3,
    /// Direction (not necessarily normalized).
    pub direction: FfiVec3,
}

impl Default for FfiRay {
    fn default() -> Self {
        Self {
            origin: FfiVec3::ZERO,
            direction: FfiVec3::new(0.0, 0.0, -1.0),
        }
    }
}

/// Entity reference in C-compatible layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiEntity {
    /// Entity ID.
    pub id: u64,
    /// Generation counter for validity checking.
    pub generation: u32,
    /// Reserved.
    pub _reserved: u32,
}

impl FfiEntity {
    /// Create an invalid entity.
    pub const INVALID: Self = Self {
        id: 0,
        generation: 0,
        _reserved: 0,
    };

    /// Returns true if this entity reference is valid.
    pub fn is_valid(&self) -> bool {
        self.id != 0
    }
}

impl Default for FfiEntity {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Component reference in C-compatible layout.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiComponent {
    /// Type identifier of the component.
    pub type_id: u64,
    /// Pointer to the component data (borrowed, not owned).
    pub data: *const c_void,
    /// Size of the component data in bytes.
    pub size: usize,
}

impl Default for FfiComponent {
    fn default() -> Self {
        Self {
            type_id: 0,
            data: ptr::null(),
            size: 0,
        }
    }
}

/// Mesh data in C-compatible layout (for passing mesh buffers across FFI).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMeshData {
    /// Vertex positions (3 floats per vertex).
    pub positions: *const f32,
    /// Vertex normals (3 floats per vertex, or null).
    pub normals: *const f32,
    /// Texture coordinates (2 floats per vertex, or null).
    pub texcoords: *const f32,
    /// Tangent vectors (4 floats per vertex, or null).
    pub tangents: *const f32,
    /// Vertex colors (4 floats per vertex, or null).
    pub colors: *const f32,
    /// Index buffer (u32 indices).
    pub indices: *const u32,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Number of indices.
    pub index_count: u32,
    /// Bounding box min.
    pub bounds_min: FfiVec3,
    /// Bounding box max.
    pub bounds_max: FfiVec3,
}

impl Default for FfiMeshData {
    fn default() -> Self {
        Self {
            positions: ptr::null(),
            normals: ptr::null(),
            texcoords: ptr::null(),
            tangents: ptr::null(),
            colors: ptr::null(),
            indices: ptr::null(),
            vertex_count: 0,
            index_count: 0,
            bounds_min: FfiVec3::ZERO,
            bounds_max: FfiVec3::ZERO,
        }
    }
}

/// Texture data in C-compatible layout.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiTextureData {
    /// Pixel data pointer.
    pub data: *const u8,
    /// Total data size in bytes.
    pub data_size: usize,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of color channels (1, 2, 3, or 4).
    pub channels: u32,
    /// Bits per channel (8, 16, or 32).
    pub bits_per_channel: u32,
    /// Whether the data is in sRGB color space (1 = sRGB, 0 = linear).
    pub is_srgb: u8,
    /// Whether the texture has mipmaps.
    pub has_mipmaps: u8,
    /// Mipmap count (1 = no mipmaps).
    pub mip_count: u16,
    /// Reserved.
    pub _reserved: [u8; 4],
}

impl Default for FfiTextureData {
    fn default() -> Self {
        Self {
            data: ptr::null(),
            data_size: 0,
            width: 0,
            height: 0,
            channels: 4,
            bits_per_channel: 8,
            is_srgb: 1,
            has_mipmaps: 0,
            mip_count: 1,
            _reserved: [0; 4],
        }
    }
}

/// Audio data in C-compatible layout.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiAudioData {
    /// PCM sample data pointer.
    pub data: *const f32,
    /// Number of samples (per channel).
    pub sample_count: u64,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Total duration in seconds.
    pub duration: f32,
    /// Bits per sample (typically 16 or 32).
    pub bits_per_sample: u32,
}

impl Default for FfiAudioData {
    fn default() -> Self {
        Self {
            data: ptr::null(),
            sample_count: 0,
            channels: 2,
            sample_rate: 48000,
            duration: 0.0,
            bits_per_sample: 32,
        }
    }
}

// ============================================================================
// Exception safety: Result -> error code conversion
// ============================================================================

/// Error code mapping for C++ exception-style error handling.
///
/// C++ consumers can check the returned error code and, on failure,
/// call `genovo_get_last_error()` to retrieve a human-readable description.
/// This avoids the need for C++ exceptions while providing rich error info.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiErrorCategory {
    /// No error.
    None = 0,
    /// Invalid argument.
    InvalidArgument = 1,
    /// Resource not found.
    NotFound = 2,
    /// Out of memory.
    OutOfMemory = 3,
    /// I/O error.
    IoError = 4,
    /// Permission denied.
    PermissionDenied = 5,
    /// Operation timed out.
    Timeout = 6,
    /// Internal engine error.
    Internal = 7,
    /// Feature not supported.
    NotSupported = 8,
    /// Operation was cancelled.
    Cancelled = 9,
}

/// Extended error information for C++ consumers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiErrorInfo {
    /// Error code (matches FfiResult).
    pub code: FfiResult,
    /// Error category.
    pub category: FfiErrorCategory,
    /// Source file name (static, valid for process lifetime).
    pub file: *const c_char,
    /// Source line number.
    pub line: u32,
    /// Reserved.
    pub _reserved: u32,
}

impl Default for FfiErrorInfo {
    fn default() -> Self {
        Self {
            code: FFI_OK,
            category: FfiErrorCategory::None,
            file: ptr::null(),
            line: 0,
            _reserved: 0,
        }
    }
}

/// Get extended error information for the last error on this thread.
///
/// # Safety
///
/// `out_info` must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_get_error_info(out_info: *mut FfiErrorInfo) -> FfiResult {
    if out_info.is_null() {
        return FFI_ERR_NULL_POINTER;
    }
    // Check if there is a last error
    let has_error = crate::genovo_get_last_error() != ptr::null();
    unsafe {
        if has_error {
            (*out_info).code = FFI_ERR_INTERNAL;
            (*out_info).category = FfiErrorCategory::Internal;
        } else {
            *out_info = FfiErrorInfo::default();
        }
    }
    FFI_OK
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- FfiString tests ----

    #[test]
    fn test_ffi_string_empty() {
        let s = FfiString::new();
        assert!(s.is_empty());
        assert_eq!(s.to_str(), "");
    }

    #[test]
    fn test_ffi_string_from_str() {
        let s = FfiString::from_str("hello world");
        assert!(!s.is_empty());
        assert_eq!(s.length, 11);
        assert_eq!(s.to_str(), "hello world");
    }

    #[test]
    fn test_ffi_string_clone() {
        let s = FfiString::from_str("test");
        let s2 = s.clone();
        assert_eq!(s.to_str(), s2.to_str());
    }

    #[test]
    fn test_ffi_string_from_cstr() {
        let cstr = CString::new("hello").unwrap();
        let s = unsafe { FfiString::from_cstr(cstr.as_ptr()) };
        assert_eq!(s.to_str(), "hello");
    }

    #[test]
    fn test_ffi_string_from_null() {
        let s = unsafe { FfiString::from_cstr(ptr::null()) };
        assert!(s.is_empty());
    }

    #[test]
    fn test_ffi_string_ffi_functions() {
        unsafe {
            let cstr = CString::new("test string").unwrap();
            let mut s = ffi_string_create(cstr.as_ptr());
            assert_eq!(ffi_string_length(&s), 11);
            let data = ffi_string_data(&s);
            assert!(!data.is_null());
            ffi_string_free(&mut s);
            assert_eq!(s.length, 0);
        }
    }

    // ---- FfiArray tests ----

    #[test]
    fn test_ffi_array_empty() {
        let arr = FfiArray::new(4);
        assert!(arr.is_empty());
        assert_eq!(arr.element_size, 4);
    }

    #[test]
    fn test_ffi_array_from_f32_vec() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let arr = FfiArray::from_f32_vec(v);
        assert_eq!(arr.length, 4);
        assert_eq!(arr.element_size, 4);
        assert_eq!(arr.byte_size(), 16);
    }

    #[test]
    fn test_ffi_array_ffi_functions() {
        let arr = ffi_array_create(8);
        assert_eq!(ffi_array_length(&arr), 0);
        assert_eq!(ffi_array_data(&arr), ptr::null());
    }

    // ---- FfiCallback tests ----

    #[test]
    fn test_ffi_callback_null() {
        let cb = FfiCallback::null();
        assert!(!cb.is_set());
    }

    #[test]
    fn test_ffi_callback_set() {
        unsafe extern "C" fn test_cb(_user_data: *mut c_void, _data: *const c_void) {}
        let cb = FfiCallback {
            func: Some(test_cb),
            user_data: ptr::null_mut(),
        };
        assert!(cb.is_set());
    }

    #[test]
    fn test_ffi_callback_invoke() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static CALLED: AtomicBool = AtomicBool::new(false);

        unsafe extern "C" fn test_cb(_user_data: *mut c_void, _data: *const c_void) {
            CALLED.store(true, Ordering::SeqCst);
        }

        CALLED.store(false, Ordering::SeqCst);
        let cb = FfiCallback {
            func: Some(test_cb),
            user_data: ptr::null_mut(),
        };
        unsafe {
            cb.invoke(ptr::null());
        }
        assert!(CALLED.load(Ordering::SeqCst));
    }

    // ---- FfiEventSystem tests ----

    #[test]
    fn test_event_system_create_destroy() {
        unsafe {
            let system = ffi_event_system_create();
            assert!(!system.is_null());
            let result = ffi_event_system_destroy(system);
            assert_eq!(result, FFI_OK);
        }
    }

    #[test]
    fn test_event_system_subscribe_unsubscribe() {
        let mut system = FfiEventSystem::new();
        let cb = FfiCallback::null();
        let sub = system.subscribe(FfiEventType::FrameBegin, cb);
        assert_ne!(sub, 0);
        assert_eq!(system.subscriber_count(FfiEventType::FrameBegin), 1);

        assert!(system.unsubscribe(sub));
        assert_eq!(system.subscriber_count(FfiEventType::FrameBegin), 0);
    }

    #[test]
    fn test_event_system_fire() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static FIRE_COUNT: AtomicU32 = AtomicU32::new(0);

        unsafe extern "C" fn count_cb(_: *mut c_void, _: *const c_void) {
            FIRE_COUNT.fetch_add(1, Ordering::SeqCst);
        }

        FIRE_COUNT.store(0, Ordering::SeqCst);
        let mut system = FfiEventSystem::new();
        let cb = FfiCallback {
            func: Some(count_cb),
            user_data: ptr::null_mut(),
        };
        system.subscribe(FfiEventType::FrameEnd, cb);
        system.subscribe(FfiEventType::FrameEnd, cb);

        unsafe {
            system.fire(FfiEventType::FrameEnd, ptr::null());
        }
        assert_eq!(FIRE_COUNT.load(Ordering::SeqCst), 2);
    }

    // ---- Cross-language type tests ----

    #[test]
    fn test_ffi_mat4_identity() {
        let m = FfiMat4::IDENTITY;
        assert_eq!(m.at(0, 0), 1.0);
        assert_eq!(m.at(1, 1), 1.0);
        assert_eq!(m.at(2, 2), 1.0);
        assert_eq!(m.at(3, 3), 1.0);
        assert_eq!(m.at(0, 1), 0.0);
    }

    #[test]
    fn test_ffi_mat4_set_get() {
        let mut m = FfiMat4::ZERO;
        m.set(1, 2, 42.0);
        assert_eq!(m.at(1, 2), 42.0);
    }

    #[test]
    fn test_ffi_rect() {
        let r = FfiRect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(r.x, 10.0);
        assert_eq!(r.y, 20.0);
        assert_eq!(r.width, 100.0);
        assert_eq!(r.height, 50.0);
    }

    #[test]
    fn test_ffi_color_constants() {
        assert_eq!(FfiColor::WHITE, FfiColor::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(FfiColor::BLACK, FfiColor::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(FfiColor::TRANSPARENT, FfiColor::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_ffi_entity_invalid() {
        let e = FfiEntity::INVALID;
        assert!(!e.is_valid());
    }

    #[test]
    fn test_ffi_ray_default() {
        let r = FfiRay::default();
        assert_eq!(r.origin, FfiVec3::ZERO);
        assert_eq!(r.direction, FfiVec3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn test_ffi_type_info() {
        unsafe {
            let mut info = FfiTypeInfo {
                name: ptr::null(),
                size: 0,
                alignment: 0,
                type_id: 0,
                is_copy: 0,
                is_send: 0,
                is_sync: 0,
                _reserved: 0,
            };
            let result = ffi_get_type_info(1, &mut info);
            assert_eq!(result, FFI_OK);
            assert_eq!(info.type_id, 1);
            assert!(info.size > 0);

            // Unknown type
            let result = ffi_get_type_info(999, &mut info);
            assert_eq!(result, FFI_ERR_INVALID_PARAMETER);
        }
    }

    // ---- Error info tests ----

    #[test]
    fn test_ffi_error_category_values() {
        assert_eq!(FfiErrorCategory::None as i32, 0);
        assert_eq!(FfiErrorCategory::InvalidArgument as i32, 1);
        assert_eq!(FfiErrorCategory::Internal as i32, 7);
    }

    #[test]
    fn test_ffi_error_info_default() {
        let info = FfiErrorInfo::default();
        assert_eq!(info.code, FFI_OK);
        assert_eq!(info.category, FfiErrorCategory::None);
    }

    // ---- RC handle tests ----

    #[test]
    fn test_ffi_rc_handle_null() {
        let h = FfiRcHandle::null();
        assert!(!h.is_valid());
    }

    #[test]
    fn test_ffi_rc_release_null() {
        unsafe {
            let result = ffi_rc_release(FfiRcHandle::null());
            assert_eq!(result, FFI_OK);
        }
    }
}
