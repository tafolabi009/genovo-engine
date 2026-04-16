//! Audio backend implementations.
//!
//! Provides pluggable audio output backends.  Each backend handles
//! platform-specific audio device initialisation, buffer submission, and
//! voice management.
//!
//! The [`NullBackend`] discards all samples and is used for headless /
//! testing scenarios.  Platform backends (WASAPI, CoreAudio, ALSA) provide
//! clear interfaces but require native libraries not linked here.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Backend error
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("failed to initialize audio device: {0}")]
    InitializationFailed(String),

    #[error("audio device lost")]
    DeviceLost,

    #[error("unsupported configuration: {0}")]
    UnsupportedConfig(String),

    #[error("backend not available on this platform")]
    NotAvailable,
}

pub type BackendResult<T> = Result<T, BackendError>;

// ---------------------------------------------------------------------------
// Voice handle
// ---------------------------------------------------------------------------

/// Opaque handle to a hardware voice managed by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoiceHandle(pub(crate) u64);

// ---------------------------------------------------------------------------
// Audio device info
// ---------------------------------------------------------------------------

/// Information about an available audio output device.
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// Supported sample rates.
    pub sample_rates: Vec<u32>,
    /// Maximum number of output channels.
    pub max_channels: u16,
    /// Whether this is the system default device.
    pub is_default: bool,
}

// ---------------------------------------------------------------------------
// AudioBackend trait
// ---------------------------------------------------------------------------

/// Low-level audio output backend interface.
///
/// Implementations handle platform-specific audio device initialisation,
/// buffer submission, and voice management.
pub trait AudioBackend: Send + Sync {
    /// Human-readable name of this backend (e.g. "WASAPI", "CoreAudio").
    fn name(&self) -> &str;

    /// Initialize the backend and open the default audio device.
    fn init(&mut self, sample_rate: u32, channels: u16, buffer_size: u32) -> BackendResult<()>;

    /// Shut down the backend and release all resources.
    fn shutdown(&mut self);

    /// Enumerate available audio output devices.
    fn enumerate_devices(&self) -> BackendResult<Vec<AudioDeviceInfo>>;

    /// Create a new voice for audio playback.
    fn create_voice(
        &mut self,
        sample_rate: u32,
        channels: u16,
    ) -> BackendResult<VoiceHandle>;

    /// Destroy a voice and free its resources.
    fn destroy_voice(&mut self, handle: VoiceHandle) -> BackendResult<()>;

    /// Submit a buffer of interleaved F32 samples to a voice for playback.
    fn submit_buffer(
        &mut self,
        handle: VoiceHandle,
        samples: &[f32],
    ) -> BackendResult<()>;

    /// Start playback of a voice.
    fn start_voice(&mut self, handle: VoiceHandle) -> BackendResult<()>;

    /// Stop playback of a voice.
    fn stop_voice(&mut self, handle: VoiceHandle) -> BackendResult<()>;

    /// Set the volume of a voice [0.0, 1.0].
    fn set_voice_volume(&mut self, handle: VoiceHandle, volume: f32) -> BackendResult<()>;

    /// Query whether the backend is currently running.
    fn is_running(&self) -> bool;

    /// Check whether this backend is available on the current platform.
    fn is_available(&self) -> bool;
}

// ===========================================================================
// Null Backend (headless / testing)
// ===========================================================================

/// A backend that discards all audio output.
///
/// Useful for automated testing, headless servers, and CI environments where
/// no audio hardware is available.
pub struct NullBackend {
    running: bool,
    sample_rate: u32,
    channels: u16,
    buffer_size: u32,
    next_voice_id: u64,
    /// All submitted buffers are accumulated here for test inspection.
    submitted_samples: Vec<f32>,
    /// Total number of frames submitted across all voices.
    total_frames_submitted: u64,
}

impl NullBackend {
    pub fn new() -> Self {
        Self {
            running: false,
            sample_rate: 0,
            channels: 0,
            buffer_size: 0,
            next_voice_id: 1,
            submitted_samples: Vec::new(),
            total_frames_submitted: 0,
        }
    }

    /// For testing: read back all samples submitted since the last clear.
    pub fn submitted_samples(&self) -> &[f32] {
        &self.submitted_samples
    }

    /// For testing: total frames submitted.
    pub fn total_frames_submitted(&self) -> u64 {
        self.total_frames_submitted
    }

    /// Clear the submitted sample buffer.
    pub fn clear_submitted(&mut self) {
        self.submitted_samples.clear();
    }
}

impl Default for NullBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioBackend for NullBackend {
    fn name(&self) -> &str {
        "Null (silent)"
    }

    fn init(&mut self, sample_rate: u32, channels: u16, buffer_size: u32) -> BackendResult<()> {
        self.sample_rate = sample_rate;
        self.channels = channels;
        self.buffer_size = buffer_size;
        self.running = true;
        log::info!(
            "NullBackend initialized: {}Hz, {} ch, buf={}",
            sample_rate,
            channels,
            buffer_size,
        );
        Ok(())
    }

    fn shutdown(&mut self) {
        self.running = false;
        self.submitted_samples.clear();
        log::info!("NullBackend shut down");
    }

    fn enumerate_devices(&self) -> BackendResult<Vec<AudioDeviceInfo>> {
        Ok(vec![AudioDeviceInfo {
            name: "Null Device".into(),
            sample_rates: vec![44100, 48000],
            max_channels: 2,
            is_default: true,
        }])
    }

    fn create_voice(&mut self, _sample_rate: u32, _channels: u16) -> BackendResult<VoiceHandle> {
        let handle = VoiceHandle(self.next_voice_id);
        self.next_voice_id += 1;
        Ok(handle)
    }

    fn destroy_voice(&mut self, _handle: VoiceHandle) -> BackendResult<()> {
        Ok(())
    }

    fn submit_buffer(&mut self, _handle: VoiceHandle, samples: &[f32]) -> BackendResult<()> {
        self.submitted_samples.extend_from_slice(samples);
        if self.channels > 0 {
            self.total_frames_submitted += samples.len() as u64 / self.channels as u64;
        }
        Ok(())
    }

    fn start_voice(&mut self, _handle: VoiceHandle) -> BackendResult<()> {
        Ok(())
    }

    fn stop_voice(&mut self, _handle: VoiceHandle) -> BackendResult<()> {
        Ok(())
    }

    fn set_voice_volume(&mut self, _handle: VoiceHandle, _volume: f32) -> BackendResult<()> {
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running
    }

    fn is_available(&self) -> bool {
        true // always available
    }
}

// ===========================================================================
// FMOD Backend (stub -- requires FMOD SDK)
// ===========================================================================

/// FMOD Studio backend.
///
/// Wraps FMOD Studio and FMOD Core via C++ FFI for professional-grade audio
/// with built-in DSP, events, and spatial features.
pub struct FmodBackend {
    _private: (),
}

impl FmodBackend {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl AudioBackend for FmodBackend {
    fn name(&self) -> &str { "FMOD Studio" }
    fn init(&mut self, _sr: u32, _ch: u16, _bs: u32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn shutdown(&mut self) {}
    fn enumerate_devices(&self) -> BackendResult<Vec<AudioDeviceInfo>> { Err(BackendError::NotAvailable) }
    fn create_voice(&mut self, _sr: u32, _ch: u16) -> BackendResult<VoiceHandle> { Err(BackendError::NotAvailable) }
    fn destroy_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn submit_buffer(&mut self, _h: VoiceHandle, _s: &[f32]) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn start_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn stop_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn set_voice_volume(&mut self, _h: VoiceHandle, _v: f32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn is_running(&self) -> bool { false }
    fn is_available(&self) -> bool { false }
}

// ===========================================================================
// Wwise Backend (stub -- requires Wwise SDK)
// ===========================================================================

/// Audiokinetic Wwise backend.
pub struct WwiseBackend {
    _private: (),
}

impl WwiseBackend {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl AudioBackend for WwiseBackend {
    fn name(&self) -> &str { "Wwise" }
    fn init(&mut self, _sr: u32, _ch: u16, _bs: u32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn shutdown(&mut self) {}
    fn enumerate_devices(&self) -> BackendResult<Vec<AudioDeviceInfo>> { Err(BackendError::NotAvailable) }
    fn create_voice(&mut self, _sr: u32, _ch: u16) -> BackendResult<VoiceHandle> { Err(BackendError::NotAvailable) }
    fn destroy_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn submit_buffer(&mut self, _h: VoiceHandle, _s: &[f32]) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn start_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn stop_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn set_voice_volume(&mut self, _h: VoiceHandle, _v: f32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn is_running(&self) -> bool { false }
    fn is_available(&self) -> bool { false }
}

// ===========================================================================
// Platform-native backends (interface stubs)
// ===========================================================================

/// WASAPI backend for Windows.
///
/// Uses Windows Audio Session API for low-latency exclusive-mode output.
/// Requires linking against Windows COM APIs at runtime.
pub struct WasapiBackend {
    _private: (),
}

impl WasapiBackend {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl AudioBackend for WasapiBackend {
    fn name(&self) -> &str { "WASAPI" }
    fn init(&mut self, _sr: u32, _ch: u16, _bs: u32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn shutdown(&mut self) {}
    fn enumerate_devices(&self) -> BackendResult<Vec<AudioDeviceInfo>> { Err(BackendError::NotAvailable) }
    fn create_voice(&mut self, _sr: u32, _ch: u16) -> BackendResult<VoiceHandle> { Err(BackendError::NotAvailable) }
    fn destroy_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn submit_buffer(&mut self, _h: VoiceHandle, _s: &[f32]) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn start_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn stop_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn set_voice_volume(&mut self, _h: VoiceHandle, _v: f32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn is_running(&self) -> bool { false }
    fn is_available(&self) -> bool { cfg!(target_os = "windows") }
}

/// Core Audio backend for macOS / iOS.
///
/// Uses Audio Units / AVAudioEngine for low-latency output on Apple platforms.
pub struct CoreAudioBackend {
    _private: (),
}

impl CoreAudioBackend {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl AudioBackend for CoreAudioBackend {
    fn name(&self) -> &str { "Core Audio" }
    fn init(&mut self, _sr: u32, _ch: u16, _bs: u32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn shutdown(&mut self) {}
    fn enumerate_devices(&self) -> BackendResult<Vec<AudioDeviceInfo>> { Err(BackendError::NotAvailable) }
    fn create_voice(&mut self, _sr: u32, _ch: u16) -> BackendResult<VoiceHandle> { Err(BackendError::NotAvailable) }
    fn destroy_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn submit_buffer(&mut self, _h: VoiceHandle, _s: &[f32]) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn start_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn stop_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn set_voice_volume(&mut self, _h: VoiceHandle, _v: f32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn is_running(&self) -> bool { false }
    fn is_available(&self) -> bool { cfg!(target_os = "macos") || cfg!(target_os = "ios") }
}

/// ALSA backend for Linux.
///
/// Uses the Advanced Linux Sound Architecture for audio output.
pub struct AlsaBackend {
    _private: (),
}

impl AlsaBackend {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl AudioBackend for AlsaBackend {
    fn name(&self) -> &str { "ALSA" }
    fn init(&mut self, _sr: u32, _ch: u16, _bs: u32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn shutdown(&mut self) {}
    fn enumerate_devices(&self) -> BackendResult<Vec<AudioDeviceInfo>> { Err(BackendError::NotAvailable) }
    fn create_voice(&mut self, _sr: u32, _ch: u16) -> BackendResult<VoiceHandle> { Err(BackendError::NotAvailable) }
    fn destroy_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn submit_buffer(&mut self, _h: VoiceHandle, _s: &[f32]) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn start_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn stop_voice(&mut self, _h: VoiceHandle) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn set_voice_volume(&mut self, _h: VoiceHandle, _v: f32) -> BackendResult<()> { Err(BackendError::NotAvailable) }
    fn is_running(&self) -> bool { false }
    fn is_available(&self) -> bool { cfg!(target_os = "linux") }
}

// ===========================================================================
// Backend selection
// ===========================================================================

/// Returns all known audio backends ordered by preference.
pub fn available_backends() -> Vec<Box<dyn AudioBackend>> {
    vec![
        Box::new(FmodBackend::new()),
        Box::new(WwiseBackend::new()),
        #[cfg(target_os = "windows")]
        Box::new(WasapiBackend::new()),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        Box::new(CoreAudioBackend::new()),
        #[cfg(target_os = "linux")]
        Box::new(AlsaBackend::new()),
    ]
}

/// Select the best available audio backend for the current platform.
pub fn select_best_backend() -> Option<Box<dyn AudioBackend>> {
    available_backends().into_iter().find(|b| b.is_available())
}

/// Create a [`NullBackend`] -- always available, discards output.
pub fn null_backend() -> NullBackend {
    NullBackend::new()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_backend_lifecycle() {
        let mut backend = NullBackend::new();
        assert!(!backend.is_running());

        backend.init(44100, 2, 512).unwrap();
        assert!(backend.is_running());
        assert!(backend.is_available());
        assert_eq!(backend.name(), "Null (silent)");

        let devices = backend.enumerate_devices().unwrap();
        assert_eq!(devices.len(), 1);
        assert!(devices[0].is_default);

        let voice = backend.create_voice(44100, 2).unwrap();
        backend.start_voice(voice).unwrap();

        let samples = vec![0.5f32; 1024];
        backend.submit_buffer(voice, &samples).unwrap();
        assert_eq!(backend.submitted_samples().len(), 1024);
        assert_eq!(backend.total_frames_submitted(), 512); // 1024 samples / 2 channels

        backend.stop_voice(voice).unwrap();
        backend.destroy_voice(voice).unwrap();

        backend.shutdown();
        assert!(!backend.is_running());
        assert!(backend.submitted_samples().is_empty());
    }

    #[test]
    fn null_backend_clear_submitted() {
        let mut backend = NullBackend::new();
        backend.init(44100, 2, 256).unwrap();
        let voice = backend.create_voice(44100, 2).unwrap();
        backend.submit_buffer(voice, &[1.0; 100]).unwrap();
        assert_eq!(backend.submitted_samples().len(), 100);
        backend.clear_submitted();
        assert!(backend.submitted_samples().is_empty());
    }

    #[test]
    fn null_backend_volume() {
        let mut backend = NullBackend::new();
        backend.init(44100, 2, 256).unwrap();
        let voice = backend.create_voice(44100, 2).unwrap();
        assert!(backend.set_voice_volume(voice, 0.5).is_ok());
    }

    #[test]
    fn fmod_not_available() {
        let backend = FmodBackend::new();
        assert!(!backend.is_available());
    }

    #[test]
    fn wwise_not_available() {
        let backend = WwiseBackend::new();
        assert!(!backend.is_available());
    }

    #[test]
    fn select_best_backend_returns_something_or_none() {
        // On any platform, this should at least not panic.
        let _backend = select_best_backend();
    }

    #[test]
    fn null_backend_always_available() {
        let backend = null_backend();
        assert!(backend.is_available());
    }
}
