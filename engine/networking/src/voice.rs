//! Voice chat framework for the Genovo networking module.
//!
//! Provides real-time voice communication over UDP with:
//!
//! - **Mu-law codec** — logarithmic 8-bit compression of 16-bit PCM audio,
//!   following ITU-T G.711 mu-law for compact, low-latency voice encoding.
//! - **Voice Activity Detection (VAD)** — energy-based silence detection to
//!   avoid transmitting dead air.
//! - **Comfort noise generation** — synthetic background noise during silence
//!   to avoid jarring silence gaps.
//! - **Jitter buffer** — reorder out-of-sequence packets and smooth playback
//!   at configurable depth (60-200 ms).
//! - **Packet loss concealment** — repeat last decoded frame with decay.
//! - **Voice channels** — team chat, proximity chat (distance-based volume),
//!   and global broadcast.
//! - **VoiceChatManager** — central coordinator for channels, mute state,
//!   and per-player volume.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default sample rate for voice audio (narrowband telephony).
pub const VOICE_SAMPLE_RATE: u32 = 8000;

/// Frame duration in milliseconds (20 ms is standard for VoIP).
pub const FRAME_DURATION_MS: u32 = 20;

/// Number of samples per frame at the default sample rate.
pub const SAMPLES_PER_FRAME: usize = (VOICE_SAMPLE_RATE * FRAME_DURATION_MS / 1000) as usize;

/// Maximum jitter buffer depth in frames.
pub const MAX_JITTER_DEPTH_FRAMES: usize = 10;

/// Default jitter buffer depth in frames (~100 ms at 20 ms/frame).
pub const DEFAULT_JITTER_DEPTH: usize = 5;

/// Mu-law compression parameter (ITU-T G.711).
pub const MU: f32 = 255.0;

/// Minimum energy threshold for Voice Activity Detection (VAD).
pub const DEFAULT_VAD_THRESHOLD: f32 = 0.005;

/// Comfort noise amplitude level.
pub const COMFORT_NOISE_LEVEL: f32 = 0.002;

/// Packet loss concealment decay factor per repeated frame.
pub const PLC_DECAY: f32 = 0.85;

/// Maximum number of consecutive concealment frames before silence.
pub const MAX_PLC_FRAMES: u32 = 5;

/// Maximum voice packet payload size in bytes.
pub const MAX_VOICE_PAYLOAD: usize = 512;

/// Proximity chat maximum audible distance (in world units).
pub const DEFAULT_MAX_PROXIMITY_DISTANCE: f32 = 50.0;

/// Minimum proximity distance (full volume).
pub const DEFAULT_MIN_PROXIMITY_DISTANCE: f32 = 2.0;

// ---------------------------------------------------------------------------
// VoiceCodec trait
// ---------------------------------------------------------------------------

/// Trait for voice audio codecs.
///
/// Implementors compress PCM f32 samples into a compact byte representation
/// and decompress back. Codecs should be designed for low-latency, real-time
/// use with frame sizes around 20 ms.
pub trait VoiceCodec: Send + Sync {
    /// Encode PCM f32 samples (range [-1, 1]) into compressed bytes.
    fn encode(&self, pcm: &[f32]) -> Vec<u8>;

    /// Decode compressed bytes back into PCM f32 samples.
    fn decode(&self, data: &[u8]) -> Vec<f32>;

    /// Return the name of this codec for logging and negotiation.
    fn name(&self) -> &str;

    /// Expected number of bytes per frame of `SAMPLES_PER_FRAME` samples.
    fn bytes_per_frame(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Mu-law codec
// ---------------------------------------------------------------------------

/// Encode a single f32 sample (in [-1, 1]) to mu-law compressed u8.
///
/// Implements ITU-T G.711 mu-law companding:
///   compressed = sgn(x) * ln(1 + mu * |x|) / ln(1 + mu)
///
/// The output is mapped to [0, 255] with 128 as the zero crossing.
pub fn mu_law_encode(sample: f32) -> u8 {
    let clamped = sample.clamp(-1.0, 1.0);
    let sign = if clamped >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let magnitude = clamped.abs();

    // Apply mu-law compression formula
    let compressed = sign * (1.0 + MU * magnitude).ln() / (1.0 + MU).ln();

    // Map from [-1, 1] to [0, 255]
    let scaled = ((compressed + 1.0) * 0.5 * 255.0).round() as u8;
    scaled
}

/// Decode a mu-law compressed u8 back to f32 sample in [-1, 1].
///
/// Inverse of mu-law encoding:
///   x = sgn(y) * (1/mu) * ((1 + mu)^|y| - 1)
pub fn mu_law_decode(byte: u8) -> f32 {
    // Map from [0, 255] to [-1, 1]
    let normalized = (byte as f32 / 255.0) * 2.0 - 1.0;

    let sign = if normalized >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let magnitude = normalized.abs();

    // Apply mu-law expansion formula
    let expanded = sign * (1.0 / MU) * (((1.0 + MU).powf(magnitude)) - 1.0);
    expanded.clamp(-1.0, 1.0)
}

/// Encode a block of PCM f32 samples to mu-law bytes.
pub fn mu_law_encode_block(samples: &[f32]) -> Vec<u8> {
    samples.iter().map(|&s| mu_law_encode(s)).collect()
}

/// Decode a block of mu-law bytes to PCM f32 samples.
pub fn mu_law_decode_block(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| mu_law_decode(b)).collect()
}

/// Simple mu-law codec implementing the `VoiceCodec` trait.
///
/// Provides 8-bit logarithmic compression of 16-bit-quality audio.
/// Compression ratio is 2:1 (16-bit -> 8-bit per sample).
pub struct SimpleCodec;

impl SimpleCodec {
    /// Create a new mu-law codec instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimpleCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceCodec for SimpleCodec {
    fn encode(&self, pcm: &[f32]) -> Vec<u8> {
        mu_law_encode_block(pcm)
    }

    fn decode(&self, data: &[u8]) -> Vec<f32> {
        mu_law_decode_block(data)
    }

    fn name(&self) -> &str {
        "mu-law-g711"
    }

    fn bytes_per_frame(&self) -> usize {
        SAMPLES_PER_FRAME
    }
}

// ---------------------------------------------------------------------------
// Voice Activity Detection (VAD)
// ---------------------------------------------------------------------------

/// Simple energy-based Voice Activity Detection.
///
/// Calculates the RMS energy of an audio frame and compares it against a
/// configurable threshold. Includes a hold timer to avoid cutting off speech
/// during brief pauses (e.g. between words).
pub struct VoiceActivityDetector {
    /// Energy threshold below which audio is considered silence.
    pub threshold: f32,
    /// How many consecutive silent frames before we declare silence.
    pub hold_frames: u32,
    /// Counter of consecutive silent frames observed.
    silent_frame_count: u32,
    /// Whether the detector currently considers the input as active speech.
    is_active: bool,
    /// Smoothed energy level (exponential moving average).
    smoothed_energy: f32,
    /// Smoothing factor for energy averaging (0..1, higher = less smoothing).
    pub smoothing_alpha: f32,
}

impl VoiceActivityDetector {
    /// Create a new VAD with the given energy threshold.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            hold_frames: 15, // hold for ~300 ms at 20 ms frames
            silent_frame_count: 0,
            is_active: false,
            smoothed_energy: 0.0,
            smoothing_alpha: 0.3,
        }
    }

    /// Create a VAD with default threshold.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_VAD_THRESHOLD)
    }

    /// Calculate RMS energy of an audio frame.
    pub fn calculate_energy(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }

    /// Process one frame and return whether voice activity is detected.
    pub fn process_frame(&mut self, samples: &[f32]) -> bool {
        let energy = Self::calculate_energy(samples);

        // Smooth the energy measurement to reduce false triggers
        self.smoothed_energy =
            self.smoothing_alpha * energy + (1.0 - self.smoothing_alpha) * self.smoothed_energy;

        if self.smoothed_energy >= self.threshold {
            self.is_active = true;
            self.silent_frame_count = 0;
        } else {
            self.silent_frame_count += 1;
            if self.silent_frame_count >= self.hold_frames {
                self.is_active = false;
            }
        }

        self.is_active
    }

    /// Query whether the VAD currently considers the input active.
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Get the current smoothed energy level.
    pub fn current_energy(&self) -> f32 {
        self.smoothed_energy
    }

    /// Reset the detector to its initial state.
    pub fn reset(&mut self) {
        self.is_active = false;
        self.silent_frame_count = 0;
        self.smoothed_energy = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Comfort Noise Generator
// ---------------------------------------------------------------------------

/// Generates low-level pseudo-random noise to fill silence periods.
///
/// During periods where the VAD determines no speech is present, comfort
/// noise prevents jarring silence. The noise level is very low and
/// shaped to match typical background ambience.
pub struct ComfortNoiseGenerator {
    /// Amplitude of the generated noise.
    pub level: f32,
    /// PRNG state (xorshift32).
    rng_state: u32,
    /// Whether comfort noise is enabled.
    pub enabled: bool,
}

impl ComfortNoiseGenerator {
    /// Create a new comfort noise generator with the given amplitude.
    pub fn new(level: f32) -> Self {
        Self {
            level,
            rng_state: 0xDEAD_BEEF,
            enabled: true,
        }
    }

    /// Create with default comfort noise level.
    pub fn with_defaults() -> Self {
        Self::new(COMFORT_NOISE_LEVEL)
    }

    /// Generate a single pseudo-random sample using xorshift32.
    fn next_sample(&mut self) -> f32 {
        // xorshift32 PRNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 17;
        self.rng_state ^= self.rng_state << 5;
        // Map u32 to [-1, 1]
        let normalized = (self.rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        normalized * self.level
    }

    /// Fill a buffer with comfort noise samples.
    pub fn generate(&mut self, output: &mut [f32]) {
        if !self.enabled {
            for sample in output.iter_mut() {
                *sample = 0.0;
            }
            return;
        }
        for sample in output.iter_mut() {
            *sample = self.next_sample();
        }
    }

    /// Mix comfort noise into an existing buffer (additive).
    pub fn mix_into(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            return;
        }
        for sample in buffer.iter_mut() {
            *sample += self.next_sample();
        }
    }

    /// Reset the PRNG to a known state (for deterministic testing).
    pub fn reset(&mut self) {
        self.rng_state = 0xDEAD_BEEF;
    }
}

// ---------------------------------------------------------------------------
// Voice Capture (stub)
// ---------------------------------------------------------------------------

/// State of the capture device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureState {
    /// Capture device is not initialised.
    Uninitialised,
    /// Capture device is ready but not recording.
    Ready,
    /// Actively recording audio.
    Recording,
    /// An error occurred with the capture device.
    Error,
}

/// Voice capture interface (platform-specific implementation required).
///
/// This provides the abstract interface for collecting microphone audio.
/// Actual audio capture is platform-dependent (WASAPI on Windows, CoreAudio
/// on macOS, ALSA/PulseAudio on Linux, etc.).
pub struct VoiceCapture {
    /// Current state of the capture device.
    pub state: CaptureState,
    /// Sample rate of the capture device.
    pub sample_rate: u32,
    /// Number of audio channels (typically 1 for voice).
    pub channels: u16,
    /// Internal buffer for accumulated samples.
    buffer: Vec<f32>,
    /// Capacity of the internal buffer in samples.
    buffer_capacity: usize,
    /// Gain applied to captured audio (1.0 = unity).
    pub input_gain: f32,
    /// Whether noise gate is active.
    pub noise_gate_enabled: bool,
    /// Noise gate threshold.
    pub noise_gate_threshold: f32,
}

impl VoiceCapture {
    /// Create a new voice capture interface.
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        let buffer_capacity = (sample_rate as usize) * 2; // 2-second buffer
        Self {
            state: CaptureState::Uninitialised,
            sample_rate,
            channels,
            buffer: Vec::with_capacity(buffer_capacity),
            buffer_capacity,
            input_gain: 1.0,
            noise_gate_enabled: false,
            noise_gate_threshold: 0.001,
        }
    }

    /// Create with default voice settings (8kHz mono).
    pub fn with_defaults() -> Self {
        Self::new(VOICE_SAMPLE_RATE, 1)
    }

    /// Initialise the capture device (stub — platform-specific).
    pub fn initialise(&mut self) -> Result<(), VoiceChatError> {
        // In a real implementation this would open the platform audio device.
        self.state = CaptureState::Ready;
        Ok(())
    }

    /// Start recording.
    pub fn start_recording(&mut self) -> Result<(), VoiceChatError> {
        if self.state != CaptureState::Ready {
            return Err(VoiceChatError::InvalidState(
                "capture device not ready".into(),
            ));
        }
        self.state = CaptureState::Recording;
        Ok(())
    }

    /// Stop recording.
    pub fn stop_recording(&mut self) -> Result<(), VoiceChatError> {
        if self.state != CaptureState::Recording {
            return Err(VoiceChatError::InvalidState(
                "capture device not recording".into(),
            ));
        }
        self.state = CaptureState::Ready;
        Ok(())
    }

    /// Push samples from the platform audio callback.
    ///
    /// In a real implementation, the platform audio thread would call this
    /// with microphone data at regular intervals.
    pub fn push_samples(&mut self, samples: &[f32]) {
        if self.state != CaptureState::Recording {
            return;
        }
        for &s in samples {
            let amplified = s * self.input_gain;
            let gated = if self.noise_gate_enabled && amplified.abs() < self.noise_gate_threshold {
                0.0
            } else {
                amplified
            };
            if self.buffer.len() < self.buffer_capacity {
                self.buffer.push(gated);
            }
        }
    }

    /// Read a frame of samples from the capture buffer.
    ///
    /// Returns `None` if not enough samples are available.
    pub fn read_frame(&mut self, frame_size: usize) -> Option<Vec<f32>> {
        if self.buffer.len() < frame_size {
            return None;
        }
        let frame: Vec<f32> = self.buffer.drain(..frame_size).collect();
        Some(frame)
    }

    /// Return the number of samples currently buffered.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    /// Shut down the capture device.
    pub fn shutdown(&mut self) {
        self.buffer.clear();
        self.state = CaptureState::Uninitialised;
    }
}

// ---------------------------------------------------------------------------
// Voice packet
// ---------------------------------------------------------------------------

/// Unique identifier for a voice chat participant.
pub type VoicePlayerId = u64;

/// A voice data packet transmitted over the network.
#[derive(Debug, Clone)]
pub struct VoicePacket {
    /// Sender player ID.
    pub player_id: VoicePlayerId,
    /// Monotonically increasing sequence number for ordering.
    pub sequence: u32,
    /// Timestamp when the packet was created (ms since session start).
    pub timestamp_ms: u32,
    /// Channel this voice packet belongs to.
    pub channel_id: u32,
    /// Encoded audio payload.
    pub payload: Vec<u8>,
    /// Whether this packet contains comfort noise rather than speech.
    pub is_comfort_noise: bool,
}

impl VoicePacket {
    /// Serialise the packet into bytes for transmission.
    pub fn serialise(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20 + self.payload.len());
        // Player ID (8 bytes)
        buf.extend_from_slice(&self.player_id.to_le_bytes());
        // Sequence (4 bytes)
        buf.extend_from_slice(&self.sequence.to_le_bytes());
        // Timestamp (4 bytes)
        buf.extend_from_slice(&self.timestamp_ms.to_le_bytes());
        // Channel (4 bytes)
        buf.extend_from_slice(&self.channel_id.to_le_bytes());
        // Flags (1 byte)
        let flags: u8 = if self.is_comfort_noise { 1 } else { 0 };
        buf.push(flags);
        // Payload length (2 bytes)
        buf.extend_from_slice(&(self.payload.len() as u16).to_le_bytes());
        // Payload
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Deserialise a voice packet from bytes.
    pub fn deserialise(data: &[u8]) -> Result<Self, VoiceChatError> {
        if data.len() < 23 {
            return Err(VoiceChatError::MalformedPacket(
                "voice packet too short".into(),
            ));
        }
        let player_id = u64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| VoiceChatError::MalformedPacket("bad player_id".into()))?,
        );
        let sequence = u32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| VoiceChatError::MalformedPacket("bad sequence".into()))?,
        );
        let timestamp_ms = u32::from_le_bytes(
            data[12..16]
                .try_into()
                .map_err(|_| VoiceChatError::MalformedPacket("bad timestamp".into()))?,
        );
        let channel_id = u32::from_le_bytes(
            data[16..20]
                .try_into()
                .map_err(|_| VoiceChatError::MalformedPacket("bad channel_id".into()))?,
        );
        let flags = data[20];
        let is_comfort_noise = (flags & 1) != 0;
        let payload_len = u16::from_le_bytes(
            data[21..23]
                .try_into()
                .map_err(|_| VoiceChatError::MalformedPacket("bad payload_len".into()))?,
        ) as usize;

        if data.len() < 23 + payload_len {
            return Err(VoiceChatError::MalformedPacket(
                "payload truncated".into(),
            ));
        }
        let payload = data[23..23 + payload_len].to_vec();

        Ok(Self {
            player_id,
            sequence,
            timestamp_ms,
            channel_id,
            payload,
            is_comfort_noise,
        })
    }
}

// ---------------------------------------------------------------------------
// Jitter Buffer
// ---------------------------------------------------------------------------

/// Entry in the jitter buffer, holding one decoded or encoded frame.
#[derive(Debug, Clone)]
struct JitterEntry {
    /// Sequence number of this entry.
    sequence: u32,
    /// Decoded PCM samples for this frame.
    samples: Vec<f32>,
    /// Whether this frame is comfort noise.
    is_comfort_noise: bool,
}

/// Ring-buffer-style jitter buffer for reordering and smoothing voice playback.
///
/// Incoming voice packets may arrive out of order or with variable delay.
/// The jitter buffer collects frames and reorders them by sequence number,
/// providing a steady stream of audio frames to the decoder.
///
/// Configurable depth (60-200 ms) controls the trade-off between latency
/// and robustness to jitter.
pub struct JitterBuffer {
    /// Buffered frames sorted by sequence number.
    entries: VecDeque<JitterEntry>,
    /// Maximum number of frames to buffer.
    pub max_depth: usize,
    /// Next expected sequence number for playback.
    next_playback_seq: u32,
    /// Whether the buffer has started receiving data.
    started: bool,
    /// Total number of packets received.
    pub packets_received: u64,
    /// Total number of packets that arrived too late (dropped).
    pub packets_too_late: u64,
    /// Total number of packets dropped due to buffer overflow.
    pub packets_overflow: u64,
    /// Minimum delay frames before playback starts.
    pub prebuffer_frames: usize,
}

impl JitterBuffer {
    /// Create a new jitter buffer with the specified depth in frames.
    ///
    /// `depth` is the maximum number of frames to buffer. At 20 ms per frame:
    /// - 3 frames = 60 ms
    /// - 5 frames = 100 ms (default)
    /// - 10 frames = 200 ms
    pub fn new(depth: usize) -> Self {
        let depth = depth.clamp(3, MAX_JITTER_DEPTH_FRAMES);
        Self {
            entries: VecDeque::with_capacity(depth),
            max_depth: depth,
            next_playback_seq: 0,
            started: false,
            packets_received: 0,
            packets_too_late: 0,
            packets_overflow: 0,
            prebuffer_frames: depth / 2,
        }
    }

    /// Create with default depth.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_JITTER_DEPTH)
    }

    /// Insert a decoded frame into the jitter buffer.
    ///
    /// Frames are inserted in sequence-number order. Late or duplicate
    /// packets are discarded.
    pub fn insert(&mut self, sequence: u32, samples: Vec<f32>, is_comfort_noise: bool) {
        self.packets_received += 1;

        if !self.started {
            self.next_playback_seq = sequence;
            self.started = true;
        }

        // Discard packets that are too old (already played or far behind)
        if self.started && sequence_before(sequence, self.next_playback_seq) {
            self.packets_too_late += 1;
            return;
        }

        // Check for buffer overflow
        if self.entries.len() >= self.max_depth {
            self.packets_overflow += 1;
            // Drop the oldest entry to make room
            self.entries.pop_front();
        }

        // Insert in sorted order by sequence number
        let insert_pos = self
            .entries
            .iter()
            .position(|e| sequence_before(sequence, e.sequence))
            .unwrap_or(self.entries.len());

        // Check for duplicate
        if insert_pos < self.entries.len() && self.entries[insert_pos].sequence == sequence {
            return; // duplicate, discard
        }
        if insert_pos > 0 {
            if let Some(prev) = self.entries.get(insert_pos - 1) {
                if prev.sequence == sequence {
                    return; // duplicate
                }
            }
        }

        self.entries.insert(
            insert_pos,
            JitterEntry {
                sequence,
                samples,
                is_comfort_noise,
            },
        );
    }

    /// Pop the next frame for playback.
    ///
    /// Returns `Some(samples)` if the next expected frame is available,
    /// `None` if the buffer is empty or prebuffering.
    pub fn pop_frame(&mut self) -> Option<Vec<f32>> {
        // Wait for prebuffer to fill
        if self.entries.len() < self.prebuffer_frames && self.prebuffer_frames > 0 {
            return None;
        }

        if let Some(front) = self.entries.front() {
            if front.sequence == self.next_playback_seq {
                let entry = self.entries.pop_front().unwrap();
                self.next_playback_seq = self.next_playback_seq.wrapping_add(1);
                Some(entry.samples)
            } else if sequence_before(self.next_playback_seq, front.sequence) {
                // Missing frame — caller should use PLC
                self.next_playback_seq = self.next_playback_seq.wrapping_add(1);
                None
            } else {
                // Stale entry, skip it
                self.entries.pop_front();
                None
            }
        } else {
            None
        }
    }

    /// Return the number of frames currently buffered.
    pub fn buffered_frames(&self) -> usize {
        self.entries.len()
    }

    /// Clear the buffer and reset state.
    pub fn reset(&mut self) {
        self.entries.clear();
        self.started = false;
        self.next_playback_seq = 0;
    }

    /// Return jitter buffer statistics.
    pub fn stats(&self) -> JitterBufferStats {
        JitterBufferStats {
            buffered_frames: self.entries.len(),
            max_depth: self.max_depth,
            packets_received: self.packets_received,
            packets_too_late: self.packets_too_late,
            packets_overflow: self.packets_overflow,
        }
    }
}

/// Helper: returns true if `a` comes before `b` with wrapping sequence numbers.
fn sequence_before(a: u32, b: u32) -> bool {
    let diff = b.wrapping_sub(a);
    diff > 0 && diff < 0x8000_0000
}

/// Statistics from the jitter buffer.
#[derive(Debug, Clone)]
pub struct JitterBufferStats {
    pub buffered_frames: usize,
    pub max_depth: usize,
    pub packets_received: u64,
    pub packets_too_late: u64,
    pub packets_overflow: u64,
}

// ---------------------------------------------------------------------------
// Packet Loss Concealment (PLC)
// ---------------------------------------------------------------------------

/// Simple packet loss concealment by repeating the last good frame with decay.
///
/// When a frame is missing from the jitter buffer, the PLC repeats the
/// previous frame but attenuates it over time to avoid artefacts.
pub struct PacketLossConcealer {
    /// The last successfully decoded frame.
    last_good_frame: Vec<f32>,
    /// Number of consecutive concealment frames generated.
    consecutive_losses: u32,
    /// Maximum concealment frames before switching to silence.
    pub max_concealment_frames: u32,
    /// Decay factor applied per repeated frame.
    pub decay_factor: f32,
    /// Total frames concealed (lifetime counter).
    pub total_concealed: u64,
}

impl PacketLossConcealer {
    /// Create a new PLC instance.
    pub fn new() -> Self {
        Self {
            last_good_frame: vec![0.0; SAMPLES_PER_FRAME],
            consecutive_losses: 0,
            max_concealment_frames: MAX_PLC_FRAMES,
            decay_factor: PLC_DECAY,
            total_concealed: 0,
        }
    }

    /// Record a good frame (reset concealment state).
    pub fn record_good_frame(&mut self, frame: &[f32]) {
        self.last_good_frame = frame.to_vec();
        self.consecutive_losses = 0;
    }

    /// Generate a concealment frame when a packet is lost.
    ///
    /// Returns a copy of the last good frame attenuated by the decay factor
    /// raised to the power of consecutive losses. Returns silence after
    /// `max_concealment_frames` consecutive losses.
    pub fn conceal(&mut self) -> Vec<f32> {
        self.consecutive_losses += 1;
        self.total_concealed += 1;

        if self.consecutive_losses > self.max_concealment_frames {
            // Too many losses — return silence
            return vec![0.0; self.last_good_frame.len()];
        }

        let attenuation = self.decay_factor.powi(self.consecutive_losses as i32);
        self.last_good_frame
            .iter()
            .map(|&s| s * attenuation)
            .collect()
    }

    /// Return whether we are currently concealing.
    pub fn is_concealing(&self) -> bool {
        self.consecutive_losses > 0
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.last_good_frame = vec![0.0; SAMPLES_PER_FRAME];
        self.consecutive_losses = 0;
    }
}

impl Default for PacketLossConcealer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VoiceTransmitter
// ---------------------------------------------------------------------------

/// Encodes captured audio, applies VAD, packetises, and transmits voice data.
///
/// The transmitter reads frames from a `VoiceCapture`, encodes them with the
/// configured codec, checks for voice activity, and produces `VoicePacket`s
/// ready for UDP transmission.
pub struct VoiceTransmitter {
    /// Codec used for encoding audio.
    codec: Box<dyn VoiceCodec>,
    /// Voice activity detector.
    pub vad: VoiceActivityDetector,
    /// Comfort noise generator for silent periods.
    pub comfort_noise: ComfortNoiseGenerator,
    /// Current packet sequence number.
    sequence: u32,
    /// Channel ID for outgoing packets.
    pub channel_id: u32,
    /// Local player ID.
    pub player_id: VoicePlayerId,
    /// Time reference for timestamps.
    start_time: Instant,
    /// Whether transmission is currently enabled.
    pub enabled: bool,
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Total frames suppressed by VAD.
    pub frames_suppressed: u64,
    /// Output packet queue.
    outgoing: VecDeque<VoicePacket>,
}

impl VoiceTransmitter {
    /// Create a new voice transmitter for the given player.
    pub fn new(player_id: VoicePlayerId, codec: Box<dyn VoiceCodec>) -> Self {
        Self {
            codec,
            vad: VoiceActivityDetector::with_defaults(),
            comfort_noise: ComfortNoiseGenerator::with_defaults(),
            sequence: 0,
            channel_id: 0,
            player_id,
            start_time: Instant::now(),
            enabled: true,
            frames_encoded: 0,
            frames_suppressed: 0,
            outgoing: VecDeque::new(),
        }
    }

    /// Set the channel ID for outgoing packets.
    pub fn set_channel(&mut self, channel_id: u32) {
        self.channel_id = channel_id;
    }

    /// Process a frame of captured audio.
    ///
    /// Applies VAD, encodes if voice is active, generates comfort noise
    /// packets during silence, and queues the resulting packet.
    pub fn process_frame(&mut self, samples: &[f32]) {
        if !self.enabled {
            return;
        }

        let is_voice = self.vad.process_frame(samples);
        let elapsed = self.start_time.elapsed();
        let timestamp_ms = elapsed.as_millis() as u32;

        if is_voice {
            // Encode the audio frame
            let encoded = self.codec.encode(samples);
            let packet = VoicePacket {
                player_id: self.player_id,
                sequence: self.sequence,
                timestamp_ms,
                channel_id: self.channel_id,
                payload: encoded,
                is_comfort_noise: false,
            };
            self.outgoing.push_back(packet);
            self.sequence = self.sequence.wrapping_add(1);
            self.frames_encoded += 1;
        } else {
            // Generate comfort noise packet periodically (every 10th silent frame)
            self.frames_suppressed += 1;
            if self.frames_suppressed % 10 == 1 {
                let mut noise = vec![0.0; samples.len()];
                self.comfort_noise.generate(&mut noise);
                let encoded = self.codec.encode(&noise);
                let packet = VoicePacket {
                    player_id: self.player_id,
                    sequence: self.sequence,
                    timestamp_ms,
                    channel_id: self.channel_id,
                    payload: encoded,
                    is_comfort_noise: true,
                };
                self.outgoing.push_back(packet);
                self.sequence = self.sequence.wrapping_add(1);
            }
        }
    }

    /// Drain all queued outgoing packets.
    pub fn drain_packets(&mut self) -> Vec<VoicePacket> {
        self.outgoing.drain(..).collect()
    }

    /// Peek at the number of queued packets.
    pub fn queued_packets(&self) -> usize {
        self.outgoing.len()
    }

    /// Reset transmitter state.
    pub fn reset(&mut self) {
        self.sequence = 0;
        self.frames_encoded = 0;
        self.frames_suppressed = 0;
        self.outgoing.clear();
        self.vad.reset();
        self.comfort_noise.reset();
        self.start_time = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// VoiceReceiver
// ---------------------------------------------------------------------------

/// Per-player receive state tracked by the `VoiceReceiver`.
struct PlayerReceiveState {
    /// Jitter buffer for this player's packets.
    jitter_buffer: JitterBuffer,
    /// Packet loss concealer.
    plc: PacketLossConcealer,
    /// Volume multiplier for this player.
    volume: f32,
    /// Whether this player is muted locally.
    muted: bool,
    /// Last decoded frame (for PLC reference).
    last_frame: Vec<f32>,
}

impl PlayerReceiveState {
    fn new(jitter_depth: usize) -> Self {
        Self {
            jitter_buffer: JitterBuffer::new(jitter_depth),
            plc: PacketLossConcealer::new(),
            volume: 1.0,
            muted: false,
            last_frame: vec![0.0; SAMPLES_PER_FRAME],
        }
    }
}

/// Receives voice packets, reorders via jitter buffer, decodes, and mixes output.
///
/// Tracks per-player state including jitter buffers and packet loss concealment.
/// The mixed output can be fed directly to the audio output device.
pub struct VoiceReceiver {
    /// Codec used for decoding audio.
    codec: Box<dyn VoiceCodec>,
    /// Per-player receive state.
    players: HashMap<VoicePlayerId, PlayerReceiveState>,
    /// Default jitter buffer depth for new players.
    pub default_jitter_depth: usize,
    /// Master volume for voice output.
    pub master_volume: f32,
    /// Mixed output buffer (reused across calls).
    mix_buffer: Vec<f32>,
}

impl VoiceReceiver {
    /// Create a new voice receiver with the given codec.
    pub fn new(codec: Box<dyn VoiceCodec>) -> Self {
        Self {
            codec,
            players: HashMap::new(),
            default_jitter_depth: DEFAULT_JITTER_DEPTH,
            master_volume: 1.0,
            mix_buffer: vec![0.0; SAMPLES_PER_FRAME],
        }
    }

    /// Receive a voice packet from the network.
    ///
    /// The packet is decoded and inserted into the appropriate player's
    /// jitter buffer.
    pub fn receive_packet(&mut self, packet: &VoicePacket) {
        let state = self
            .players
            .entry(packet.player_id)
            .or_insert_with(|| PlayerReceiveState::new(self.default_jitter_depth));

        if state.muted {
            return;
        }

        // Decode the payload
        let decoded = self.codec.decode(&packet.payload);
        state
            .jitter_buffer
            .insert(packet.sequence, decoded, packet.is_comfort_noise);
    }

    /// Mix one frame of audio from all active players into the output buffer.
    ///
    /// Returns a reference to the mixed samples. Missing frames trigger
    /// packet loss concealment.
    pub fn mix_frame(&mut self) -> Vec<f32> {
        let frame_size = SAMPLES_PER_FRAME;
        self.mix_buffer.clear();
        self.mix_buffer.resize(frame_size, 0.0);

        let player_ids: Vec<VoicePlayerId> = self.players.keys().copied().collect();

        for pid in player_ids {
            let state = self.players.get_mut(&pid).unwrap();
            if state.muted {
                continue;
            }

            let frame = if let Some(frame) = state.jitter_buffer.pop_frame() {
                state.plc.record_good_frame(&frame);
                state.last_frame = frame.clone();
                frame
            } else {
                // Packet loss — use concealment
                state.plc.conceal()
            };

            // Mix into output with volume scaling
            let vol = state.volume * self.master_volume;
            for (i, &sample) in frame.iter().enumerate() {
                if i < self.mix_buffer.len() {
                    self.mix_buffer[i] += sample * vol;
                }
            }
        }

        // Clip the output to [-1, 1]
        for sample in self.mix_buffer.iter_mut() {
            *sample = sample.clamp(-1.0, 1.0);
        }

        self.mix_buffer.clone()
    }

    /// Set the volume for a specific player.
    pub fn set_player_volume(&mut self, player_id: VoicePlayerId, volume: f32) {
        if let Some(state) = self.players.get_mut(&player_id) {
            state.volume = volume.max(0.0);
        }
    }

    /// Mute or unmute a specific player.
    pub fn set_player_muted(&mut self, player_id: VoicePlayerId, muted: bool) {
        if let Some(state) = self.players.get_mut(&player_id) {
            state.muted = muted;
        }
    }

    /// Check if a player is muted.
    pub fn is_player_muted(&self, player_id: VoicePlayerId) -> bool {
        self.players
            .get(&player_id)
            .map(|s| s.muted)
            .unwrap_or(false)
    }

    /// Remove a player from the receiver (e.g., they disconnected).
    pub fn remove_player(&mut self, player_id: VoicePlayerId) {
        self.players.remove(&player_id);
    }

    /// Get jitter buffer stats for a specific player.
    pub fn player_stats(&self, player_id: VoicePlayerId) -> Option<JitterBufferStats> {
        self.players.get(&player_id).map(|s| s.jitter_buffer.stats())
    }

    /// Get the number of active players.
    pub fn active_player_count(&self) -> usize {
        self.players.len()
    }

    /// Reset the receiver, clearing all player state.
    pub fn reset(&mut self) {
        self.players.clear();
        self.mix_buffer = vec![0.0; SAMPLES_PER_FRAME];
    }
}

// ---------------------------------------------------------------------------
// Voice channel types
// ---------------------------------------------------------------------------

/// Type of voice channel, determining routing and volume behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoiceChannelType {
    /// Global broadcast — all players hear each other at full volume.
    Global,
    /// Team-only chat — only team members hear each other.
    Team,
    /// Proximity-based — volume attenuates with distance between players.
    Proximity,
    /// Private channel — explicit player list.
    Private,
}

/// A voice channel that groups players for communication.
///
/// Channels control who can hear whom and how volume is affected.
/// Proximity channels additionally scale volume based on the 3D distance
/// between the speaking and listening players.
pub struct VoiceChannel {
    /// Unique identifier for this channel.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Type of channel (global, team, proximity, private).
    pub channel_type: VoiceChannelType,
    /// Players currently in this channel.
    pub members: Vec<VoicePlayerId>,
    /// Maximum number of members (0 = unlimited).
    pub max_members: usize,
    /// Whether the channel is currently active.
    pub active: bool,
    /// Volume multiplier for the entire channel.
    pub volume: f32,
    /// For proximity channels: maximum audible distance.
    pub max_distance: f32,
    /// For proximity channels: distance at which volume is full.
    pub min_distance: f32,
    /// For team channels: team identifier.
    pub team_id: Option<u32>,
}

impl VoiceChannel {
    /// Create a new global voice channel.
    pub fn new_global(id: u32, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            channel_type: VoiceChannelType::Global,
            members: Vec::new(),
            max_members: 0,
            active: true,
            volume: 1.0,
            max_distance: DEFAULT_MAX_PROXIMITY_DISTANCE,
            min_distance: DEFAULT_MIN_PROXIMITY_DISTANCE,
            team_id: None,
        }
    }

    /// Create a new team voice channel.
    pub fn new_team(id: u32, name: &str, team_id: u32) -> Self {
        Self {
            id,
            name: name.to_string(),
            channel_type: VoiceChannelType::Team,
            members: Vec::new(),
            max_members: 0,
            active: true,
            volume: 1.0,
            max_distance: DEFAULT_MAX_PROXIMITY_DISTANCE,
            min_distance: DEFAULT_MIN_PROXIMITY_DISTANCE,
            team_id: Some(team_id),
        }
    }

    /// Create a new proximity voice channel.
    pub fn new_proximity(id: u32, name: &str, max_dist: f32, min_dist: f32) -> Self {
        Self {
            id,
            name: name.to_string(),
            channel_type: VoiceChannelType::Proximity,
            members: Vec::new(),
            max_members: 0,
            active: true,
            volume: 1.0,
            max_distance: max_dist,
            min_distance: min_dist,
            team_id: None,
        }
    }

    /// Create a new private voice channel.
    pub fn new_private(id: u32, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            channel_type: VoiceChannelType::Private,
            members: Vec::new(),
            max_members: 0,
            active: true,
            volume: 1.0,
            max_distance: DEFAULT_MAX_PROXIMITY_DISTANCE,
            min_distance: DEFAULT_MIN_PROXIMITY_DISTANCE,
            team_id: None,
        }
    }

    /// Add a player to this channel.
    pub fn add_member(&mut self, player_id: VoicePlayerId) -> Result<(), VoiceChatError> {
        if self.members.contains(&player_id) {
            return Err(VoiceChatError::AlreadyInChannel(player_id, self.id));
        }
        if self.max_members > 0 && self.members.len() >= self.max_members {
            return Err(VoiceChatError::ChannelFull(self.id));
        }
        self.members.push(player_id);
        Ok(())
    }

    /// Remove a player from this channel.
    pub fn remove_member(&mut self, player_id: VoicePlayerId) -> bool {
        if let Some(pos) = self.members.iter().position(|&p| p == player_id) {
            self.members.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if a player is a member of this channel.
    pub fn has_member(&self, player_id: VoicePlayerId) -> bool {
        self.members.contains(&player_id)
    }

    /// Get the member count.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Calculate the volume scaling for proximity-based channels.
    ///
    /// Returns a volume multiplier in [0, 1] based on the distance between
    /// the speaker and listener. Full volume at `min_distance`, fading
    /// linearly to zero at `max_distance`.
    pub fn proximity_volume(&self, distance: f32) -> f32 {
        if self.channel_type != VoiceChannelType::Proximity {
            return self.volume;
        }
        if distance <= self.min_distance {
            return self.volume;
        }
        if distance >= self.max_distance {
            return 0.0;
        }
        let range = self.max_distance - self.min_distance;
        if range <= 0.0 {
            return self.volume;
        }
        let t = (distance - self.min_distance) / range;
        self.volume * (1.0 - t)
    }

    /// Calculate inverse-square proximity volume (more realistic falloff).
    pub fn proximity_volume_inverse_square(&self, distance: f32) -> f32 {
        if self.channel_type != VoiceChannelType::Proximity {
            return self.volume;
        }
        if distance <= self.min_distance {
            return self.volume;
        }
        if distance >= self.max_distance {
            return 0.0;
        }
        let ref_dist = self.min_distance.max(0.01);
        let atten = (ref_dist / distance).powi(2);
        self.volume * atten.min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Player position (for proximity chat)
// ---------------------------------------------------------------------------

/// 3D position of a player for proximity-based voice chat.
#[derive(Debug, Clone, Copy)]
pub struct PlayerPosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl PlayerPosition {
    /// Create a new position.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Zero position.
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Compute the Euclidean distance to another position.
    pub fn distance_to(&self, other: &PlayerPosition) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ---------------------------------------------------------------------------
// VoiceChatError
// ---------------------------------------------------------------------------

/// Errors that can occur in the voice chat system.
#[derive(Debug, Clone)]
pub enum VoiceChatError {
    /// The system is in an invalid state for the requested operation.
    InvalidState(String),
    /// A packet could not be parsed.
    MalformedPacket(String),
    /// The specified channel does not exist.
    ChannelNotFound(u32),
    /// The player is already in the specified channel.
    AlreadyInChannel(VoicePlayerId, u32),
    /// The channel is at maximum capacity.
    ChannelFull(u32),
    /// The specified player was not found.
    PlayerNotFound(VoicePlayerId),
    /// A codec error occurred.
    CodecError(String),
    /// Network I/O error.
    NetworkError(String),
}

impl std::fmt::Display for VoiceChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidState(msg) => write!(f, "invalid state: {}", msg),
            Self::MalformedPacket(msg) => write!(f, "malformed packet: {}", msg),
            Self::ChannelNotFound(id) => write!(f, "channel {} not found", id),
            Self::AlreadyInChannel(player, ch) => {
                write!(f, "player {} already in channel {}", player, ch)
            }
            Self::ChannelFull(id) => write!(f, "channel {} is full", id),
            Self::PlayerNotFound(id) => write!(f, "player {} not found", id),
            Self::CodecError(msg) => write!(f, "codec error: {}", msg),
            Self::NetworkError(msg) => write!(f, "network error: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// VoiceChatManager
// ---------------------------------------------------------------------------

/// Per-player state maintained by the manager.
struct ManagedPlayerState {
    /// Whether the player's microphone is muted (not sending voice).
    mic_muted: bool,
    /// Whether the player is deafened (not receiving voice).
    deafened: bool,
    /// Per-player output volume override (1.0 = default).
    output_volume: f32,
    /// 3D position for proximity calculations.
    position: PlayerPosition,
    /// Voice transmitter for this player (only for the local player).
    transmitter: Option<VoiceTransmitter>,
    /// Channels this player is currently in.
    active_channels: Vec<u32>,
}

/// Central manager for the voice chat system.
///
/// Coordinates voice channels, player state, mute/unmute, volume control,
/// and proximity-based audio routing. Typically one instance exists per
/// game client.
pub struct VoiceChatManager {
    /// All registered voice channels.
    channels: HashMap<u32, VoiceChannel>,
    /// Per-player state.
    players: HashMap<VoicePlayerId, ManagedPlayerState>,
    /// Voice receiver that mixes incoming audio.
    pub receiver: VoiceReceiver,
    /// The local player's ID.
    pub local_player_id: VoicePlayerId,
    /// Next channel ID for auto-assignment.
    next_channel_id: u32,
    /// Master volume for all voice output.
    pub master_volume: f32,
    /// Whether the voice system is active.
    pub active: bool,
    /// Voice chat quality settings.
    pub quality: VoiceQuality,
}

/// Voice quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceQuality {
    /// Low quality: 8kHz, higher compression.
    Low,
    /// Medium quality: 8kHz, standard mu-law.
    Medium,
    /// High quality: 16kHz, lower compression (placeholder for future codecs).
    High,
}

impl VoiceChatManager {
    /// Create a new voice chat manager for the specified local player.
    pub fn new(local_player_id: VoicePlayerId) -> Self {
        let codec = Box::new(SimpleCodec::new());
        Self {
            channels: HashMap::new(),
            players: HashMap::new(),
            receiver: VoiceReceiver::new(codec),
            local_player_id,
            next_channel_id: 1,
            master_volume: 1.0,
            active: true,
            quality: VoiceQuality::Medium,
        }
    }

    /// Create a new voice channel and return its ID.
    pub fn create_channel(
        &mut self,
        name: &str,
        channel_type: VoiceChannelType,
    ) -> u32 {
        let id = self.next_channel_id;
        self.next_channel_id += 1;

        let channel = match channel_type {
            VoiceChannelType::Global => VoiceChannel::new_global(id, name),
            VoiceChannelType::Team => VoiceChannel::new_team(id, name, 0),
            VoiceChannelType::Proximity => VoiceChannel::new_proximity(
                id,
                name,
                DEFAULT_MAX_PROXIMITY_DISTANCE,
                DEFAULT_MIN_PROXIMITY_DISTANCE,
            ),
            VoiceChannelType::Private => VoiceChannel::new_private(id, name),
        };
        self.channels.insert(id, channel);
        id
    }

    /// Create a team channel with a specific team ID.
    pub fn create_team_channel(&mut self, name: &str, team_id: u32) -> u32 {
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        let channel = VoiceChannel::new_team(id, name, team_id);
        self.channels.insert(id, channel);
        id
    }

    /// Create a proximity channel with custom distance parameters.
    pub fn create_proximity_channel(
        &mut self,
        name: &str,
        max_distance: f32,
        min_distance: f32,
    ) -> u32 {
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        let channel = VoiceChannel::new_proximity(id, name, max_distance, min_distance);
        self.channels.insert(id, channel);
        id
    }

    /// Remove a channel by ID. All members are ejected first.
    pub fn remove_channel(&mut self, channel_id: u32) -> Result<(), VoiceChatError> {
        let channel = self
            .channels
            .get(&channel_id)
            .ok_or(VoiceChatError::ChannelNotFound(channel_id))?;
        let members = channel.members.clone();
        for pid in members {
            if let Some(ps) = self.players.get_mut(&pid) {
                ps.active_channels.retain(|&c| c != channel_id);
            }
        }
        self.channels.remove(&channel_id);
        Ok(())
    }

    /// Register a player in the voice system.
    pub fn register_player(&mut self, player_id: VoicePlayerId) {
        let is_local = player_id == self.local_player_id;
        let transmitter = if is_local {
            let codec = Box::new(SimpleCodec::new());
            Some(VoiceTransmitter::new(player_id, codec))
        } else {
            None
        };

        self.players.insert(
            player_id,
            ManagedPlayerState {
                mic_muted: false,
                deafened: false,
                output_volume: 1.0,
                position: PlayerPosition::zero(),
                transmitter,
                active_channels: Vec::new(),
            },
        );
    }

    /// Unregister a player, removing them from all channels.
    pub fn unregister_player(&mut self, player_id: VoicePlayerId) {
        // Remove from all channels
        for channel in self.channels.values_mut() {
            channel.remove_member(player_id);
        }
        self.players.remove(&player_id);
        self.receiver.remove_player(player_id);
    }

    /// Join a player to a channel.
    pub fn join_channel(
        &mut self,
        player_id: VoicePlayerId,
        channel_id: u32,
    ) -> Result<(), VoiceChatError> {
        let channel = self
            .channels
            .get_mut(&channel_id)
            .ok_or(VoiceChatError::ChannelNotFound(channel_id))?;
        channel.add_member(player_id)?;

        let ps = self
            .players
            .get_mut(&player_id)
            .ok_or(VoiceChatError::PlayerNotFound(player_id))?;
        if !ps.active_channels.contains(&channel_id) {
            ps.active_channels.push(channel_id);
        }
        Ok(())
    }

    /// Remove a player from a channel.
    pub fn leave_channel(
        &mut self,
        player_id: VoicePlayerId,
        channel_id: u32,
    ) -> Result<(), VoiceChatError> {
        let channel = self
            .channels
            .get_mut(&channel_id)
            .ok_or(VoiceChatError::ChannelNotFound(channel_id))?;
        channel.remove_member(player_id);

        if let Some(ps) = self.players.get_mut(&player_id) {
            ps.active_channels.retain(|&c| c != channel_id);
        }
        Ok(())
    }

    /// Mute/unmute a player's microphone (stops sending).
    pub fn set_mic_muted(&mut self, player_id: VoicePlayerId, muted: bool) {
        if let Some(ps) = self.players.get_mut(&player_id) {
            ps.mic_muted = muted;
            if let Some(tx) = &mut ps.transmitter {
                tx.enabled = !muted;
            }
        }
    }

    /// Deafen/undeafen a player (stops receiving).
    pub fn set_deafened(&mut self, player_id: VoicePlayerId, deafened: bool) {
        if let Some(ps) = self.players.get_mut(&player_id) {
            ps.deafened = deafened;
        }
        self.receiver.set_player_muted(player_id, deafened);
    }

    /// Set the output volume for a specific player (0.0 to 2.0).
    pub fn set_player_volume(&mut self, player_id: VoicePlayerId, volume: f32) {
        let vol = volume.clamp(0.0, 2.0);
        if let Some(ps) = self.players.get_mut(&player_id) {
            ps.output_volume = vol;
        }
        self.receiver.set_player_volume(player_id, vol);
    }

    /// Update a player's 3D position (for proximity voice channels).
    pub fn update_player_position(
        &mut self,
        player_id: VoicePlayerId,
        x: f32,
        y: f32,
        z: f32,
    ) {
        if let Some(ps) = self.players.get_mut(&player_id) {
            ps.position = PlayerPosition::new(x, y, z);
        }
    }

    /// Get the volume for a player considering proximity channels.
    ///
    /// This is called during mixing to scale audio based on distance
    /// when the speaker and listener share a proximity channel.
    pub fn calculate_proximity_volume(
        &self,
        speaker_id: VoicePlayerId,
        listener_id: VoicePlayerId,
    ) -> f32 {
        let speaker_pos = match self.players.get(&speaker_id) {
            Some(ps) => ps.position,
            None => return 0.0,
        };
        let listener_pos = match self.players.get(&listener_id) {
            Some(ps) => ps.position,
            None => return 0.0,
        };

        let distance = speaker_pos.distance_to(&listener_pos);
        let mut max_vol = 0.0_f32;

        // Check all proximity channels both players share
        for channel in self.channels.values() {
            if channel.channel_type != VoiceChannelType::Proximity {
                continue;
            }
            if !channel.has_member(speaker_id) || !channel.has_member(listener_id) {
                continue;
            }
            let vol = channel.proximity_volume(distance);
            max_vol = max_vol.max(vol);
        }

        max_vol
    }

    /// Process incoming voice packet from the network.
    pub fn process_incoming_packet(&mut self, data: &[u8]) -> Result<(), VoiceChatError> {
        let packet = VoicePacket::deserialise(data)?;

        // Check if the player is in any channel the local player shares
        let player_id = packet.player_id;
        let shares_channel = self.shares_any_channel(player_id, self.local_player_id);
        if !shares_channel {
            return Ok(()); // ignore packets from players not in our channels
        }

        // Apply proximity volume if applicable
        let prox_vol = self.calculate_proximity_volume(player_id, self.local_player_id);
        if prox_vol > 0.0 {
            self.receiver.set_player_volume(player_id, prox_vol);
        }

        self.receiver.receive_packet(&packet);
        Ok(())
    }

    /// Check if two players share any voice channel.
    fn shares_any_channel(&self, a: VoicePlayerId, b: VoicePlayerId) -> bool {
        for channel in self.channels.values() {
            if channel.has_member(a) && channel.has_member(b) {
                return true;
            }
        }
        false
    }

    /// Mix one frame of received voice audio for output.
    pub fn mix_output_frame(&mut self) -> Vec<f32> {
        self.receiver.master_volume = self.master_volume;
        self.receiver.mix_frame()
    }

    /// Get the list of channel IDs a player is in.
    pub fn player_channels(&self, player_id: VoicePlayerId) -> Vec<u32> {
        self.players
            .get(&player_id)
            .map(|ps| ps.active_channels.clone())
            .unwrap_or_default()
    }

    /// Get the number of registered channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get the number of registered players.
    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    /// Check if a player's mic is muted.
    pub fn is_mic_muted(&self, player_id: VoicePlayerId) -> bool {
        self.players
            .get(&player_id)
            .map(|ps| ps.mic_muted)
            .unwrap_or(false)
    }

    /// Check if a player is deafened.
    pub fn is_deafened(&self, player_id: VoicePlayerId) -> bool {
        self.players
            .get(&player_id)
            .map(|ps| ps.deafened)
            .unwrap_or(false)
    }

    /// Get a channel by ID.
    pub fn get_channel(&self, channel_id: u32) -> Option<&VoiceChannel> {
        self.channels.get(&channel_id)
    }

    /// Shut down the voice chat system.
    pub fn shutdown(&mut self) {
        self.active = false;
        self.channels.clear();
        for (_, ps) in self.players.iter_mut() {
            if let Some(tx) = &mut ps.transmitter {
                tx.reset();
            }
        }
        self.players.clear();
        self.receiver.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mu_law_roundtrip_zero() {
        let encoded = mu_law_encode(0.0);
        let decoded = mu_law_decode(encoded);
        assert!(decoded.abs() < 0.02, "zero roundtrip: got {}", decoded);
    }

    #[test]
    fn mu_law_roundtrip_positive() {
        for &val in &[0.1, 0.25, 0.5, 0.75, 1.0] {
            let encoded = mu_law_encode(val);
            let decoded = mu_law_decode(encoded);
            let error = (decoded - val).abs();
            assert!(
                error < 0.05,
                "mu-law roundtrip for {}: got {} (error {})",
                val,
                decoded,
                error
            );
        }
    }

    #[test]
    fn mu_law_roundtrip_negative() {
        for &val in &[-0.1, -0.5, -1.0] {
            let encoded = mu_law_encode(val);
            let decoded = mu_law_decode(encoded);
            let error = (decoded - val).abs();
            assert!(
                error < 0.05,
                "mu-law roundtrip for {}: got {} (error {})",
                val,
                decoded,
                error
            );
        }
    }

    #[test]
    fn mu_law_clamps_out_of_range() {
        let enc_high = mu_law_encode(2.0);
        let enc_one = mu_law_encode(1.0);
        assert_eq!(enc_high, enc_one);

        let enc_low = mu_law_encode(-2.0);
        let enc_neg = mu_law_encode(-1.0);
        assert_eq!(enc_low, enc_neg);
    }

    #[test]
    fn simple_codec_roundtrip() {
        let codec = SimpleCodec::new();
        let samples: Vec<f32> = (0..160).map(|i| (i as f32 / 160.0) * 2.0 - 1.0).collect();
        let encoded = codec.encode(&samples);
        assert_eq!(encoded.len(), 160);
        let decoded = codec.decode(&encoded);
        assert_eq!(decoded.len(), 160);
        for (orig, dec) in samples.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.1, "roundtrip error too large");
        }
    }

    #[test]
    fn vad_detects_silence() {
        let mut vad = VoiceActivityDetector::new(0.01);
        let silence = vec![0.0; SAMPLES_PER_FRAME];
        for _ in 0..20 {
            vad.process_frame(&silence);
        }
        assert!(!vad.is_active());
    }

    #[test]
    fn vad_detects_speech() {
        let mut vad = VoiceActivityDetector::new(0.01);
        let speech: Vec<f32> = (0..SAMPLES_PER_FRAME)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();
        let active = vad.process_frame(&speech);
        assert!(active);
    }

    #[test]
    fn jitter_buffer_in_order() {
        let mut jb = JitterBuffer::new(5);
        jb.prebuffer_frames = 0;
        for i in 0..5 {
            jb.insert(i, vec![i as f32; 10], false);
        }
        for i in 0..5 {
            let frame = jb.pop_frame().unwrap();
            assert_eq!(frame[0], i as f32);
        }
    }

    #[test]
    fn jitter_buffer_reorders() {
        let mut jb = JitterBuffer::new(5);
        jb.prebuffer_frames = 0;
        // Insert out of order
        jb.insert(2, vec![2.0; 10], false);
        jb.insert(0, vec![0.0; 10], false);
        jb.insert(1, vec![1.0; 10], false);
        let f0 = jb.pop_frame().unwrap();
        assert_eq!(f0[0], 0.0);
        let f1 = jb.pop_frame().unwrap();
        assert_eq!(f1[0], 1.0);
    }

    #[test]
    fn plc_decays() {
        let mut plc = PacketLossConcealer::new();
        let frame = vec![1.0; 10];
        plc.record_good_frame(&frame);
        let c1 = plc.conceal();
        let c2 = plc.conceal();
        assert!(c1[0] > c2[0], "second concealment should be quieter");
    }

    #[test]
    fn voice_packet_serialise_roundtrip() {
        let packet = VoicePacket {
            player_id: 42,
            sequence: 100,
            timestamp_ms: 5000,
            channel_id: 1,
            payload: vec![1, 2, 3, 4, 5],
            is_comfort_noise: false,
        };
        let data = packet.serialise();
        let decoded = VoicePacket::deserialise(&data).unwrap();
        assert_eq!(decoded.player_id, 42);
        assert_eq!(decoded.sequence, 100);
        assert_eq!(decoded.timestamp_ms, 5000);
        assert_eq!(decoded.channel_id, 1);
        assert_eq!(decoded.payload, vec![1, 2, 3, 4, 5]);
        assert!(!decoded.is_comfort_noise);
    }

    #[test]
    fn proximity_volume_scaling() {
        let ch = VoiceChannel::new_proximity(1, "prox", 50.0, 5.0);
        assert_eq!(ch.proximity_volume(0.0), 1.0);
        assert_eq!(ch.proximity_volume(5.0), 1.0);
        assert_eq!(ch.proximity_volume(50.0), 0.0);
        let mid = ch.proximity_volume(27.5);
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn voice_chat_manager_lifecycle() {
        let mut mgr = VoiceChatManager::new(1);
        mgr.register_player(1);
        mgr.register_player(2);

        let ch = mgr.create_channel("global", VoiceChannelType::Global);
        mgr.join_channel(1, ch).unwrap();
        mgr.join_channel(2, ch).unwrap();

        assert_eq!(mgr.get_channel(ch).unwrap().member_count(), 2);

        mgr.set_mic_muted(1, true);
        assert!(mgr.is_mic_muted(1));

        mgr.leave_channel(2, ch).unwrap();
        assert_eq!(mgr.get_channel(ch).unwrap().member_count(), 1);

        mgr.shutdown();
        assert_eq!(mgr.channel_count(), 0);
        assert_eq!(mgr.player_count(), 0);
    }

    #[test]
    fn comfort_noise_generates() {
        let mut cng = ComfortNoiseGenerator::with_defaults();
        let mut buf = vec![0.0; 100];
        cng.generate(&mut buf);
        let any_nonzero = buf.iter().any(|&s| s != 0.0);
        assert!(any_nonzero, "comfort noise should produce non-zero samples");
        let max = buf.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        assert!(max < 0.1, "comfort noise should be very quiet");
    }
}
