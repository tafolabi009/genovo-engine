// =============================================================================
// Genovo Engine - Platform Factory
// =============================================================================
//
// Runtime platform selection. Uses `cfg` attributes to compile the correct
// backend for the target OS. Console platforms are behind cargo features.

use crate::interface::Platform;

/// Create the platform backend for the current target OS.
///
/// This is the main entry point for the platform subsystem. The engine
/// calls this once during initialization and stores the returned
/// `Box<dyn Platform>` for the lifetime of the application.
///
/// # Panics
///
/// Panics if no platform backend is available for the current target.
pub fn create_platform() -> Box<dyn Platform> {
    log::info!(
        "Creating platform backend for: {}",
        crate::PlatformType::current()
    );

    #[cfg(target_os = "windows")]
    {
        log::info!("Initializing Windows platform (Win32 + XInput)");
        Box::new(crate::windows::WindowsPlatform::new())
    }

    #[cfg(target_os = "macos")]
    {
        log::info!("Initializing macOS platform (Cocoa + Metal view)");
        Box::new(crate::macos::MacOsPlatform::new())
    }

    #[cfg(target_os = "linux")]
    {
        // Display server (X11 vs Wayland) is auto-detected by winit and
        // reported via LinuxPlatform::display_server for engine-level queries.
        log::info!("Initializing Linux platform (X11/Wayland)");
        Box::new(crate::linux::LinuxPlatform::new())
    }

    #[cfg(target_os = "ios")]
    {
        log::info!("Initializing iOS platform");
        Box::new(crate::ios::IosPlatform::new())
    }

    #[cfg(target_os = "android")]
    {
        log::info!("Initializing Android platform (JNI)");
        Box::new(crate::android::AndroidPlatform::new())
    }

    #[cfg(feature = "xbox")]
    {
        log::info!("Initializing Xbox platform (GDK)");
        Box::new(crate::xbox::XboxPlatform::new())
    }

    #[cfg(feature = "playstation")]
    {
        log::info!("Initializing PlayStation platform");
        Box::new(crate::playstation::PlayStationPlatform::new())
    }

    #[cfg(not(any(
        target_os = "windows",
        target_os = "macos",
        target_os = "linux",
        target_os = "ios",
        target_os = "android",
        feature = "xbox",
        feature = "playstation"
    )))]
    {
        panic!(
            "No platform backend available for this target. \
             Supported targets: Windows, macOS, Linux, iOS, Android, Xbox (feature), PlayStation (feature)."
        );
    }
}
