// =============================================================================
// Genovo Engine - Android Platform Backend
// =============================================================================
//
// Android platform implementation with JNI integration, ANativeWindow,
// touch input, sensor support, and Vulkan rendering.
//
// Android uses the NDK's NativeActivity or GameActivity model. The OS owns
// the main thread (the "UI thread"); native code runs on a separate thread
// managed by ALooper. Until the JNI/NDK toolchain is configured, Platform
// trait methods return sensible defaults or PlatformError::Unsupported errors.

#![cfg(target_os = "android")]

use std::collections::HashMap;

use crate::interface::events::PlatformEvent;
use crate::interface::{
    CursorType, DisplayInfo, Platform, PlatformError, RawWindowHandle, RenderBackend, Result,
    SystemInfo, WindowDesc, WindowHandle,
};

// -----------------------------------------------------------------------------
// NDK FFI type definitions
// -----------------------------------------------------------------------------
// These type aliases represent the opaque NDK handles that a full
// implementation would use. They are defined here for documentation and
// type-safety in function signatures.

/// Opaque handle to an `ANativeActivity` from the NDK.
/// Provides access to the JNI environment, asset manager, internal/external
/// data paths, and the activity's Java-side instance.
pub type ANativeActivity = std::ffi::c_void;

/// Opaque handle to an `ANativeWindow` from the NDK.
/// Represents the rendering surface. Obtained via the `onNativeWindowCreated`
/// callback and released on `onNativeWindowDestroyed`.
pub type ANativeWindow = std::ffi::c_void;

/// Opaque handle to an `ALooper` from the NDK.
/// The main-thread looper polls for input events (AInputQueue), sensor data
/// (ASensorEventQueue), and custom pipe-based messages.
pub type ALooper = std::ffi::c_void;

/// Opaque handle to an `AInputQueue` from the NDK.
/// Delivers `AInputEvent` structs for touch (AINPUT_EVENT_TYPE_MOTION) and
/// hardware button (AINPUT_EVENT_TYPE_KEY) events.
pub type AInputQueue = std::ffi::c_void;

/// Opaque handle to an `ASensorManager` from the NDK.
/// Used to enumerate available sensors and create event queues.
pub type ASensorManager = std::ffi::c_void;

/// Opaque handle to an `ASensorEventQueue` from the NDK.
/// Delivers sensor readings (accelerometer, gyroscope, etc.) at a configured
/// sampling rate.
pub type ASensorEventQueue = std::ffi::c_void;

// -----------------------------------------------------------------------------
// NDK FFI declarations
// -----------------------------------------------------------------------------
// These extern signatures document the NDK functions that a full
// implementation would link against. They are declared but not called until
// the Android NDK toolchain is configured.

extern "C" {
    // ALooper
    // fn ALooper_pollAll(timeout_ms: i32, out_fd: *mut i32, out_events: *mut i32, out_data: *mut *mut std::ffi::c_void) -> i32;

    // ANativeWindow
    // fn ANativeWindow_getWidth(window: *mut ANativeWindow) -> i32;
    // fn ANativeWindow_getHeight(window: *mut ANativeWindow) -> i32;
    // fn ANativeWindow_acquire(window: *mut ANativeWindow);
    // fn ANativeWindow_release(window: *mut ANativeWindow);

    // ASensorManager
    // fn ASensorManager_getInstance() -> *mut ASensorManager;
    // fn ASensorManager_getDefaultSensor(manager: *mut ASensorManager, sensor_type: i32) -> *const std::ffi::c_void;
    // fn ASensorManager_createEventQueue(manager: *mut ASensorManager, looper: *mut ALooper, ident: i32, callback: *mut std::ffi::c_void, data: *mut std::ffi::c_void) -> *mut ASensorEventQueue;

    // AInputQueue
    // fn AInputQueue_getEvent(queue: *mut AInputQueue, out_event: *mut *mut std::ffi::c_void) -> i32;
    // fn AInputQueue_finishEvent(queue: *mut AInputQueue, event: *mut std::ffi::c_void, handled: i32);
}

// -----------------------------------------------------------------------------
// Android Sensor Types
// -----------------------------------------------------------------------------

/// Android sensor types available through ASensorManager.
///
/// Maps to the `ASENSOR_TYPE_*` constants defined in `<android/sensor.h>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AndroidSensorType {
    /// 3-axis accelerometer in m/s^2, including gravity.
    Accelerometer,
    /// 3-axis gyroscope in rad/s.
    Gyroscope,
    /// 3-axis magnetometer (compass) in micro-Tesla.
    MagneticField,
    /// Ambient light level in lux.
    Light,
    /// Proximity distance in cm (often binary: near/far).
    Proximity,
    /// Linear acceleration (accelerometer minus gravity) in m/s^2.
    LinearAcceleration,
    /// Rotation vector as a quaternion.
    RotationVector,
    /// Game rotation vector (no magnetic calibration).
    GameRotationVector,
    /// Step counter (cumulative since boot).
    StepCounter,
}

// -----------------------------------------------------------------------------
// Android Platform
// -----------------------------------------------------------------------------

/// Android platform implementation using JNI and the NDK.
///
/// # Architecture
///
/// The Android activity lifecycle is managed by the OS. The engine integrates
/// via `ANativeActivity` (or the newer `GameActivity` from the Android Game
/// Development Kit). Key lifecycle callbacks:
///
/// - `onNativeWindowCreated` — the rendering surface is available; store the
///   `ANativeWindow` pointer and begin rendering.
/// - `onNativeWindowDestroyed` — the surface is being torn down; stop rendering,
///   release GPU resources.
/// - `onPause` / `onResume` — pause/resume game logic and audio.
/// - `onLowMemory` — free non-essential caches.
///
/// Input arrives through `AInputQueue` (attached to the `ALooper`). Touch
/// events are `AINPUT_EVENT_TYPE_MOTION` with action codes for
/// `AMOTION_EVENT_ACTION_DOWN/MOVE/UP/CANCEL/POINTER_DOWN/POINTER_UP`.
///
/// Sensor data is read from an `ASensorEventQueue` also attached to the
/// `ALooper`. The engine polls all sources with `ALooper_pollAll(0, ...)`.
///
/// # Features
///
/// - ANativeActivity / GameActivity integration
/// - ANativeWindow for rendering surface
/// - Multi-touch input via AInputQueue
/// - Sensor access (accelerometer, gyroscope) via ASensorManager
/// - Vulkan as primary render backend (OpenGL ES fallback)
/// - Android lifecycle (onPause, onResume, onDestroy)
/// - JNI calls for clipboard, system info, vibration
/// - Immersive / edge-to-edge display mode
pub struct AndroidPlatform {
    /// The single native window (Android typically has one).
    window: Option<NativeWindowData>,
    /// Engine handle for the main window.
    window_handle_id: u64,
    /// Accumulated events.
    pending_events: Vec<PlatformEvent>,
    /// Whether the app is currently in the foreground.
    is_foreground: bool,
    /// Whether the surface is currently valid (between surfaceCreated and surfaceDestroyed).
    surface_valid: bool,
    /// Active touch pointers keyed by pointer ID from AMotionEvent.
    active_touches: HashMap<u64, (f64, f64)>,
    /// Which sensors are currently enabled.
    enabled_sensors: Vec<AndroidSensorType>,
}

/// Native Android window data.
#[derive(Debug)]
struct NativeWindowData {
    /// Pointer to ANativeWindow.
    a_native_window: *mut std::ffi::c_void,
    /// Display density (DPI / 160.0).
    scale_factor: f64,
    /// Width of the native window in pixels.
    width: u32,
    /// Height of the native window in pixels.
    height: u32,
}

unsafe impl Send for NativeWindowData {}
unsafe impl Sync for NativeWindowData {}

impl AndroidPlatform {
    /// Create a new Android platform instance.
    ///
    /// Typically called from the native activity's `ANativeActivity_onCreate`
    /// or from the GameActivity's initialization callback.
    ///
    /// Initialization sequence (when NDK toolchain is available):
    /// 1. Receive `ANativeActivity*` from the JNI entry point.
    /// 2. Prepare the main-thread `ALooper` via `ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS)`.
    /// 3. Obtain `ASensorManager` via `ASensorManager_getInstance()` and create
    ///    an event queue for accelerometer and gyroscope.
    /// 4. Request immersive sticky mode via JNI call to
    ///    `View.setSystemUiVisibility(SYSTEM_UI_FLAG_IMMERSIVE_STICKY | ...)`.
    pub fn new() -> Self {
        log::info!("AndroidPlatform created (requires NDK/JNI toolchain for full integration)");

        Self {
            window: None,
            window_handle_id: 1,
            pending_events: Vec::new(),
            is_foreground: true,
            surface_valid: false,
            active_touches: HashMap::new(),
            enabled_sensors: Vec::new(),
        }
    }

    /// Process events from ALooper (input + sensor events).
    ///
    /// Called once per frame. In a full implementation this does:
    ///
    /// 1. `ALooper_pollAll(0, ...)` — non-blocking poll of all looper sources.
    /// 2. For each `AInputEvent` from the `AInputQueue`:
    ///    - `AINPUT_EVENT_TYPE_MOTION` — extract pointer count, action, and
    ///      per-pointer (x, y, pressure) via `AMotionEvent_get*` functions.
    ///      Convert to `PlatformEvent::TouchInput`.
    ///    - `AINPUT_EVENT_TYPE_KEY` — extract key code and action via
    ///      `AKeyEvent_getKeyCode` / `AKeyEvent_getAction`. Handle Back,
    ///      Volume, etc.
    ///    - Call `AInputQueue_finishEvent` to acknowledge each event.
    /// 3. Read sensor events from `ASensorEventQueue_getEvents` and emit
    ///    them as custom sensor events or cache them for gameplay queries.
    fn pump_events(&mut self) {
        // Event processing is driven by ALooper_pollAll with timeout 0
        // (non-blocking). The looper returns identifiers for each ready source:
        //
        //   LOOPER_ID_INPUT  -> process AInputQueue events
        //   LOOPER_ID_SENSOR -> process ASensorEventQueue events
        //   LOOPER_ID_USER   -> process custom pipe messages (lifecycle)
    }

    /// Handle the ANativeWindow being created (surface available for rendering).
    ///
    /// Called from the `onNativeWindowCreated` lifecycle callback. The engine
    /// should:
    /// 1. Store the `ANativeWindow*` and call `ANativeWindow_acquire`.
    /// 2. Query width/height via `ANativeWindow_getWidth/Height`.
    /// 3. Initialize or re-initialize the Vulkan swapchain.
    /// 4. Emit a `WindowResize` event with the surface dimensions.
    pub fn on_native_window_created(&mut self, native_window: *mut std::ffi::c_void) {
        // In a full implementation:
        // ANativeWindow_acquire(native_window);
        // let w = ANativeWindow_getWidth(native_window);
        // let h = ANativeWindow_getHeight(native_window);
        self.surface_valid = true;
        log::info!("AndroidPlatform: native window created (surface valid)");

        if self.window.is_none() {
            self.window = Some(NativeWindowData {
                a_native_window: native_window,
                scale_factor: 1.0,
                width: 0,
                height: 0,
            });
        } else if let Some(ref mut data) = self.window {
            data.a_native_window = native_window;
        }
    }

    /// Handle the ANativeWindow being destroyed.
    ///
    /// Called from the `onNativeWindowDestroyed` lifecycle callback. The engine
    /// must stop rendering and release the Vulkan swapchain before returning,
    /// because the ANativeWindow is invalid after this callback returns.
    /// Call `ANativeWindow_release` on the stored pointer.
    pub fn on_native_window_destroyed(&mut self) {
        // In a full implementation:
        // ANativeWindow_release(self.window.a_native_window);
        self.surface_valid = false;
        if let Some(ref mut data) = self.window {
            data.a_native_window = std::ptr::null_mut();
        }
        log::info!("AndroidPlatform: native window destroyed (surface invalid)");
    }

    /// Trigger device vibration via JNI.
    ///
    /// Calls `android.os.Vibrator.vibrate(long milliseconds)` through JNI:
    /// 1. Obtain `JNIEnv*` from the activity.
    /// 2. Get the Vibrator service: `context.getSystemService(Context.VIBRATOR_SERVICE)`.
    /// 3. Call `vibrator.vibrate(VibrationEffect.createOneShot(duration, amplitude))`.
    ///
    /// Requires `android.permission.VIBRATE` in AndroidManifest.xml.
    pub fn vibrate(&self, duration_ms: u32) {
        log::debug!(
            "AndroidPlatform::vibrate({}ms) — requires JNI/Vibrator service",
            duration_ms
        );
    }

    /// Get display cutout / notch insets via JNI.
    ///
    /// Queries `WindowInsets.getDisplayCutout()` through JNI to obtain the
    /// safe insets (top, left, bottom, right) in pixels. On devices without
    /// a cutout, all values are zero.
    ///
    /// Requires API level 28+ (Android 9 Pie) for DisplayCutout support.
    pub fn get_display_cutout_insets(&self) -> (i32, i32, i32, i32) {
        // JNI call sequence:
        //   Window window = activity.getWindow();
        //   WindowInsets insets = window.getDecorView().getRootWindowInsets();
        //   DisplayCutout cutout = insets.getDisplayCutout();
        //   return (cutout.getSafeInsetTop(), cutout.getSafeInsetLeft(),
        //           cutout.getSafeInsetBottom(), cutout.getSafeInsetRight());
        (0, 0, 0, 0)
    }

    /// Called from the native activity's `onPause` callback.
    pub fn on_pause(&mut self) {
        self.is_foreground = false;
        self.pending_events.push(PlatformEvent::AppSuspend);
    }

    /// Called from the native activity's `onResume` callback.
    pub fn on_resume(&mut self) {
        self.is_foreground = true;
        self.pending_events.push(PlatformEvent::AppResume);
    }

    /// Called from the native activity's `onLowMemory` callback.
    pub fn on_low_memory(&mut self) {
        self.pending_events.push(PlatformEvent::AppLowMemory);
    }
}

impl Default for AndroidPlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl Platform for AndroidPlatform {
    fn create_window(&self, _desc: &WindowDesc) -> Result<WindowHandle> {
        // On Android the "window" is the ANativeWindow provided by the activity
        // lifecycle — the engine does not create OS windows. The activity's
        // `onNativeWindowCreated` callback provides the surface; the engine
        // wraps it as a WindowHandle.
        //
        // WindowDesc fields (title, width, height, decorations) are ignored
        // because Android apps are always full-screen activities.
        //
        // If the surface is already available, return the existing handle.
        if self.surface_valid && self.window.is_some() {
            return Ok(WindowHandle(self.window_handle_id));
        }
        Err(PlatformError::Unsupported(
            "Android windows are provided by the activity lifecycle, not created by the engine. \
             Wait for the onNativeWindowCreated callback."
                .into(),
        ))
    }

    fn destroy_window(&self, handle: WindowHandle) {
        // No-op: Android windows are managed by the OS activity lifecycle.
        // The ANativeWindow is released in `on_native_window_destroyed`.
        log::debug!(
            "AndroidPlatform::destroy_window is a no-op on Android (handle={})",
            handle
        );
    }

    fn poll_events(&mut self) -> Vec<PlatformEvent> {
        // Pump ALooper for input and sensor events, then drain the queue.
        self.pump_events();
        std::mem::take(&mut self.pending_events)
    }

    fn get_render_backend(&self) -> RenderBackend {
        // Vulkan is the preferred backend on Android. OpenGL ES is the fallback
        // for older devices (pre-API 24). A full implementation would check
        // Vulkan availability at runtime via `vkEnumerateInstanceVersion` or
        // by attempting `vkCreateInstance` and falling back to
        // `RenderBackend::OpenGLES` on failure.
        RenderBackend::Vulkan
    }

    fn get_display_info(&self) -> Vec<DisplayInfo> {
        // A full implementation queries display properties via JNI:
        //   WindowManager wm = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        //   Display display = wm.getDefaultDisplay();
        //   DisplayMetrics metrics = new DisplayMetrics();
        //   display.getRealMetrics(metrics);
        //   -> widthPixels, heightPixels, densityDpi, xdpi, ydpi
        //   display.getRefreshRate() -> Hz
        //
        // For now return an empty list; the renderer uses the ANativeWindow
        // dimensions directly.
        Vec::new()
    }

    fn set_cursor(&self, _cursor: CursorType) {
        // No-op on Android: no visible mouse cursor on touch devices.
        // Android 7+ does support mouse pointer icons via
        // PointerIcon.setPointerIcon, but this is uncommon for games.
    }

    fn set_clipboard(&self, text: &str) {
        // A full implementation calls via JNI:
        //   ClipboardManager cm = (ClipboardManager)
        //       context.getSystemService(Context.CLIPBOARD_SERVICE);
        //   ClipData clip = ClipData.newPlainText("genovo", text);
        //   cm.setPrimaryClip(clip);
        log::debug!(
            "AndroidPlatform::set_clipboard — requires ClipboardManager JNI call, len={}",
            text.len()
        );
    }

    fn get_clipboard(&self) -> Option<String> {
        // A full implementation calls via JNI:
        //   ClipboardManager cm = ...getSystemService(Context.CLIPBOARD_SERVICE);
        //   ClipData clip = cm.getPrimaryClip();
        //   if (clip != null && clip.getItemCount() > 0)
        //       return clip.getItemAt(0).getText().toString();
        log::debug!("AndroidPlatform::get_clipboard — requires ClipboardManager JNI call");
        None
    }

    fn get_system_info(&self) -> SystemInfo {
        // A full implementation queries via JNI and /proc:
        //   - CPU: Build.HARDWARE or /proc/cpuinfo "Hardware" line
        //   - Cores: Runtime.getRuntime().availableProcessors() or
        //            sysconf(_SC_NPROCESSORS_ONLN)
        //   - GPU: VkPhysicalDeviceProperties.deviceName from Vulkan enumeration
        //   - VRAM: VkPhysicalDeviceMemoryProperties heap sizes (device-local)
        //   - RAM: parse /proc/meminfo "MemTotal" line
        //   - OS: Build.VERSION.RELEASE for version string,
        //         Build.VERSION.SDK_INT for API level
        //   - Model: Build.MANUFACTURER + " " + Build.MODEL
        SystemInfo {
            cpu_name: String::from("ARM (query via /proc/cpuinfo or Build.HARDWARE)"),
            cpu_cores: 8, // Common for modern Android devices (4 big + 4 little)
            gpu_name: String::from("Unknown (query via VkPhysicalDeviceProperties)"),
            gpu_vram_mb: 0, // Unified memory on most mobile SoCs
            total_ram_mb: 8192, // 8 GB is a common modern default
            os_name: String::from("Android"),
            os_version: String::from("Unknown (query via Build.VERSION.RELEASE)"),
        }
    }

    fn get_raw_window_handle(&self, handle: WindowHandle) -> RawWindowHandle {
        // Returns the ANativeWindow pointer for Vulkan surface creation via
        // `vkCreateAndroidSurfaceKHR`. The pointer is valid between
        // `onNativeWindowCreated` and `onNativeWindowDestroyed`.
        if handle.0 == self.window_handle_id {
            if let Some(ref data) = self.window {
                return RawWindowHandle::Android {
                    a_native_window: data.a_native_window,
                };
            }
        }
        log::error!("get_raw_window_handle called with invalid handle: {}", handle);
        RawWindowHandle::Unavailable
    }

    fn get_window_scale_factor(&self, _handle: WindowHandle) -> f64 {
        // Returns the display density from DisplayMetrics (DPI / 160.0).
        // For example: 320 DPI -> 2.0, 480 DPI -> 3.0.
        // The cached value is set when the native window is created.
        self.window
            .as_ref()
            .map(|d| d.scale_factor)
            .unwrap_or(1.0)
    }
}
