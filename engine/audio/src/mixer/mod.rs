//! Software audio mixer.
//!
//! Provides a complete software mixing pipeline: WAV parsing, PCM sample
//! management, multi-channel voice playback with pitch shifting via linear
//! interpolation, bus routing with hierarchical volume/mute, priority-based
//! voice stealing, and fade-in/fade-out envelopes.

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors produced by audio operations.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("invalid mixer handle: {0:?}")]
    InvalidHandle(MixerHandle),

    #[error("audio clip has no sample data")]
    EmptyClip,

    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("no audio backend initialized")]
    NoBackend,

    #[error("backend error: {0}")]
    BackendError(String),

    #[error("voice limit reached (max {0})")]
    VoiceLimitReached(u32),

    #[error("WAV parse error: {0}")]
    WavParseError(String),
}

pub type AudioResult<T> = Result<T, AudioError>;

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

/// Opaque handle to a playing sound instance in the mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MixerHandle(pub u64);

// ---------------------------------------------------------------------------
// Audio format
// ---------------------------------------------------------------------------

/// PCM sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// 16-bit signed integer PCM.
    I16,
    /// 32-bit floating point PCM.
    F32,
    /// 24-bit signed integer PCM (stored in 32-bit containers).
    I24,
    /// 8-bit unsigned integer PCM.
    U8,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::F32
    }
}

impl AudioFormat {
    /// Bytes consumed per single sample in this format.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            AudioFormat::U8 => 1,
            AudioFormat::I16 => 2,
            AudioFormat::I24 => 3,
            AudioFormat::F32 => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Audio clip
// ---------------------------------------------------------------------------

/// A loaded audio asset containing normalised f32 sample data.
#[derive(Debug, Clone)]
pub struct AudioClip {
    /// Human-readable name or asset path.
    pub name: String,
    /// Normalised interleaved sample data in [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Sample rate in Hz (e.g. 44_100, 48_000).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Original sample format from the source file.
    pub format: AudioFormat,
}

impl AudioClip {
    /// Duration of the clip in seconds.
    pub fn duration(&self) -> f32 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        let frame_count = self.samples.len() / self.channels as usize;
        frame_count as f32 / self.sample_rate as f32
    }

    /// Number of audio frames (one sample per channel).
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Whether this clip is mono.
    pub fn is_mono(&self) -> bool {
        self.channels == 1
    }

    /// Total number of individual samples (all channels).
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    // -----------------------------------------------------------------------
    // WAV parser
    // -----------------------------------------------------------------------

    /// Parse a WAV file from raw bytes and return a normalised `AudioClip`.
    ///
    /// Supports PCM formats: 8-bit unsigned, 16-bit signed, 24-bit signed,
    /// and 32-bit IEEE float.  Handles the standard RIFF/WAVE container with
    /// `fmt ` and `data` chunks (possibly in any order, possibly with extra
    /// chunks in between).
    pub fn from_wav_bytes(bytes: &[u8]) -> AudioResult<AudioClip> {
        Self::from_wav_bytes_named(bytes, "unnamed")
    }

    /// Same as [`from_wav_bytes`] but lets you attach a name.
    pub fn from_wav_bytes_named(bytes: &[u8], name: &str) -> AudioResult<AudioClip> {
        if bytes.len() < 44 {
            return Err(AudioError::WavParseError("file too small for a WAV header".into()));
        }

        // -- RIFF header --
        if &bytes[0..4] != b"RIFF" {
            return Err(AudioError::WavParseError("missing RIFF magic".into()));
        }
        // bytes[4..8] = file size - 8 (we don't validate this strictly)
        if &bytes[8..12] != b"WAVE" {
            return Err(AudioError::WavParseError("missing WAVE magic".into()));
        }

        // -- Walk chunks --
        let mut pos: usize = 12;
        let mut fmt_parsed = false;
        let mut audio_format: u16 = 0;
        let mut channels: u16 = 0;
        let mut sample_rate: u32 = 0;
        let mut bits_per_sample: u16 = 0;
        let mut data_samples: Option<Vec<f32>> = None;

        while pos + 8 <= bytes.len() {
            let chunk_id = &bytes[pos..pos + 4];
            let chunk_size = u32::from_le_bytes([
                bytes[pos + 4],
                bytes[pos + 5],
                bytes[pos + 6],
                bytes[pos + 7],
            ]) as usize;
            pos += 8;

            if chunk_id == b"fmt " {
                if chunk_size < 16 || pos + 16 > bytes.len() {
                    return Err(AudioError::WavParseError("fmt chunk too small".into()));
                }
                audio_format = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]);
                channels = u16::from_le_bytes([bytes[pos + 2], bytes[pos + 3]]);
                sample_rate = u32::from_le_bytes([
                    bytes[pos + 4],
                    bytes[pos + 5],
                    bytes[pos + 6],
                    bytes[pos + 7],
                ]);
                // bytes[pos+8..pos+12] = byte rate (skip)
                // bytes[pos+12..pos+14] = block align (skip)
                bits_per_sample = u16::from_le_bytes([bytes[pos + 14], bytes[pos + 15]]);
                fmt_parsed = true;
            } else if chunk_id == b"data" {
                if !fmt_parsed {
                    return Err(AudioError::WavParseError(
                        "data chunk before fmt chunk".into(),
                    ));
                }
                let data_end = (pos + chunk_size).min(bytes.len());
                let raw = &bytes[pos..data_end];

                // PCM integer (format tag 1) or IEEE float (format tag 3)
                data_samples = Some(match (audio_format, bits_per_sample) {
                    (1, 8) => Self::decode_u8_pcm(raw),
                    (1, 16) => Self::decode_i16_pcm(raw),
                    (1, 24) => Self::decode_i24_pcm(raw),
                    (1, 32) => Self::decode_i32_pcm(raw),
                    (3, 32) => Self::decode_f32_pcm(raw),
                    _ => {
                        return Err(AudioError::UnsupportedFormat(format!(
                            "WAV format tag={audio_format}, bits={bits_per_sample}"
                        )));
                    }
                });
            }
            // Advance to next chunk (chunks are word-aligned).
            pos += (chunk_size + 1) & !1;
        }

        let samples = data_samples.ok_or_else(|| {
            AudioError::WavParseError("no data chunk found".into())
        })?;

        if samples.is_empty() {
            return Err(AudioError::EmptyClip);
        }

        let format = match (audio_format, bits_per_sample) {
            (1, 8) => AudioFormat::U8,
            (1, 16) => AudioFormat::I16,
            (1, 24) => AudioFormat::I24,
            (3, 32) | (1, 32) => AudioFormat::F32,
            _ => AudioFormat::F32,
        };

        Ok(AudioClip {
            name: name.to_string(),
            samples,
            sample_rate,
            channels,
            format,
        })
    }

    // -- Decoder helpers (private) --

    fn decode_u8_pcm(raw: &[u8]) -> Vec<f32> {
        raw.iter().map(|&b| (b as f32 / 128.0) - 1.0).collect()
    }

    fn decode_i16_pcm(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(2)
            .map(|c| {
                let s = i16::from_le_bytes([c[0], c[1]]);
                s as f32 / 32768.0
            })
            .collect()
    }

    fn decode_i24_pcm(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(3)
            .map(|c| {
                // Sign-extend the 24-bit value into 32 bits.
                let lo = c[0] as u32;
                let mid = c[1] as u32;
                let hi = c[2] as u32;
                let val = lo | (mid << 8) | (hi << 16);
                // Sign-extend: if bit 23 is set, fill upper 8 bits with 1s.
                let val = if val & 0x80_0000 != 0 {
                    (val | 0xFF00_0000) as i32
                } else {
                    val as i32
                };
                val as f32 / 8_388_608.0
            })
            .collect()
    }

    fn decode_i32_pcm(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(4)
            .map(|c| {
                let s = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                s as f32 / 2_147_483_648.0
            })
            .collect()
    }

    fn decode_f32_pcm(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Create a simple sine-wave clip (useful for testing).
    pub fn sine_wave(frequency: f32, duration: f32, sample_rate: u32) -> AudioClip {
        let frame_count = (duration * sample_rate as f32) as usize;
        let mut samples = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            let t = i as f32 / sample_rate as f32;
            samples.push((2.0 * std::f32::consts::PI * frequency * t).sin());
        }
        AudioClip {
            name: format!("sine_{frequency}hz"),
            samples,
            sample_rate,
            channels: 1,
            format: AudioFormat::F32,
        }
    }

    /// Create a silent clip of the given duration.
    pub fn silence(duration: f32, sample_rate: u32, channels: u16) -> AudioClip {
        let frame_count = (duration * sample_rate as f32) as usize;
        let total = frame_count * channels as usize;
        AudioClip {
            name: "silence".into(),
            samples: vec![0.0; total],
            sample_rate,
            channels,
            format: AudioFormat::F32,
        }
    }
}

// ---------------------------------------------------------------------------
// Audio bus
// ---------------------------------------------------------------------------

/// Logical audio bus for grouping and controlling volume of related sounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioBus {
    /// Master bus -- all audio routes through here.
    Master,
    /// Background music.
    Music,
    /// Sound effects.
    SFX,
    /// Voice / dialogue.
    Voice,
    /// Ambient / environmental sounds.
    Ambient,
    /// User-defined bus identified by index.
    Custom(u16),
}

impl Default for AudioBus {
    fn default() -> Self {
        Self::Master
    }
}

impl AudioBus {
    /// Every non-master bus routes into the Master bus.
    pub fn parent(self) -> Option<AudioBus> {
        match self {
            AudioBus::Master => None,
            _ => Some(AudioBus::Master),
        }
    }

    /// Return all built-in buses (excluding Custom).
    pub fn all_builtin() -> &'static [AudioBus] {
        &[
            AudioBus::Master,
            AudioBus::Music,
            AudioBus::SFX,
            AudioBus::Voice,
            AudioBus::Ambient,
        ]
    }
}

/// Runtime state for an audio bus.
#[derive(Debug, Clone)]
struct BusState {
    volume: f32,
    muted: bool,
    /// Temporary mix buffer (stereo interleaved) for the current update tick.
    buffer: Vec<f32>,
}

impl BusState {
    fn new() -> Self {
        Self {
            volume: 1.0,
            muted: false,
            buffer: Vec::new(),
        }
    }

    /// Effective volume considering mute.
    fn effective_volume(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.volume
        }
    }
}

// ---------------------------------------------------------------------------
// Play parameters
// ---------------------------------------------------------------------------

/// Parameters for playing an audio clip.
#[derive(Debug, Clone)]
pub struct AudioSource {
    /// The clip to play.
    pub clip_name: String,
    /// Volume multiplier [0.0, ...). Values above 1.0 amplify.
    pub volume: f32,
    /// Pitch multiplier. 1.0 = normal, 0.5 = half speed, 2.0 = double speed.
    pub pitch: f32,
    /// Whether the clip should loop.
    pub looping: bool,
    /// Whether this source is spatialized (3D positioned).
    pub spatial: bool,
    /// The bus this source is routed to.
    pub bus: AudioBus,
    /// Priority for voice management.  Higher priority voices steal lower ones
    /// when the voice limit is reached.  Range [0, 255]; 0 = lowest.
    pub priority: u8,
    /// Start time offset within the clip, in seconds.
    pub start_offset: f32,
    /// Optional fade-in duration in seconds.
    pub fade_in: f32,
    /// Pan value [-1.0 left, 0.0 centre, 1.0 right].
    pub pan: f32,
}

impl Default for AudioSource {
    fn default() -> Self {
        Self {
            clip_name: String::new(),
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            spatial: false,
            bus: AudioBus::SFX,
            priority: 128,
            start_offset: 0.0,
            fade_in: 0.0,
            pan: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Channel state
// ---------------------------------------------------------------------------

/// Playback state of an audio channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelState {
    Playing,
    Paused,
    Stopping,
    Stopped,
    FadingIn,
    FadingOut,
}

// ---------------------------------------------------------------------------
// Audio channel (voice)
// ---------------------------------------------------------------------------

/// Internal state of a playing voice in the mixer.
#[derive(Debug)]
pub struct AudioChannel {
    /// Handle identifying this voice.
    pub handle: MixerHandle,
    /// The source parameters used to start this voice.
    pub source: AudioSource,
    /// Index into the clip registry (`SoftwareMixer::clips`).
    clip_index: usize,
    /// Current playback position as a *fractional frame index*.
    /// The integer part selects the frame; the fractional part is used for
    /// linear interpolation when the pitch != 1.0.
    playback_pos: f64,
    /// Current volume (may differ from source.volume during fades).
    pub effective_volume: f32,
    /// Current state.
    pub state: ChannelState,
    /// Priority used for voice stealing.
    pub priority: u8,
    /// Pan [-1 left, +1 right].
    pub pan: f32,
    /// Fade envelope elapsed time (seconds).
    fade_elapsed: f32,
    /// Fade total duration (seconds).  Zero means no fade active.
    fade_duration: f32,
    /// Volume at the start of the current fade.
    fade_start_volume: f32,
    /// Volume at the end of the current fade.
    fade_end_volume: f32,
}

impl AudioChannel {
    fn new(
        handle: MixerHandle,
        source: AudioSource,
        clip_index: usize,
        initial_frame: f64,
    ) -> Self {
        let fade_in = source.fade_in;
        let volume = source.volume;
        let pan = source.pan;
        let priority = source.priority;
        let (state, fade_duration, fade_start, fade_end) = if fade_in > 0.0 {
            (ChannelState::FadingIn, fade_in, 0.0_f32, volume)
        } else {
            (ChannelState::Playing, 0.0_f32, volume, volume)
        };

        AudioChannel {
            handle,
            source,
            clip_index,
            playback_pos: initial_frame,
            effective_volume: fade_start,
            state,
            priority,
            pan,
            fade_elapsed: 0.0,
            fade_duration,
            fade_start_volume: fade_start,
            fade_end_volume: fade_end,
        }
    }
}

// ---------------------------------------------------------------------------
// AudioMixer trait
// ---------------------------------------------------------------------------

/// Primary audio mixing interface.
///
/// Manages voices, applies bus routing, and submits mixed audio to the
/// platform audio backend.
pub trait AudioMixer: Send + Sync {
    /// Play an audio source and return a handle for further control.
    fn play(&mut self, source: AudioSource) -> AudioResult<MixerHandle>;

    /// Stop a playing sound (optionally with a fade-out duration).
    fn stop(&mut self, handle: MixerHandle, fade_out: f32) -> AudioResult<()>;

    /// Pause a playing sound.
    fn pause(&mut self, handle: MixerHandle) -> AudioResult<()>;

    /// Resume a paused sound.
    fn resume(&mut self, handle: MixerHandle) -> AudioResult<()>;

    /// Set the volume of a playing sound.
    fn set_volume(&mut self, handle: MixerHandle, volume: f32) -> AudioResult<()>;

    /// Set the pitch/playback rate of a playing sound.
    fn set_pitch(&mut self, handle: MixerHandle, pitch: f32) -> AudioResult<()>;

    /// Set the volume for an entire bus.
    fn set_bus_volume(&mut self, bus: AudioBus, volume: f32);

    /// Mute or unmute a bus.
    fn set_bus_mute(&mut self, bus: AudioBus, muted: bool);

    /// Get the current state of a playing sound.
    fn channel_state(&self, handle: MixerHandle) -> AudioResult<ChannelState>;

    /// Stop all playing sounds.
    fn stop_all(&mut self);

    /// Pause all playing sounds.
    fn pause_all(&mut self);

    /// Resume all paused sounds.
    fn resume_all(&mut self);

    /// Update the mixer (process fades, voice stealing, submit to backend).
    /// Called once per frame.
    fn update(&mut self, dt: f32);

    /// Maximum number of simultaneous voices.
    fn max_voices(&self) -> u32;

    /// Number of currently active voices.
    fn active_voice_count(&self) -> u32;

    /// Set the master volume [0.0, 1.0].
    fn set_master_volume(&mut self, volume: f32);

    /// Get the current master volume.
    fn master_volume(&self) -> f32;
}

// ---------------------------------------------------------------------------
// Software mixer implementation
// ---------------------------------------------------------------------------

/// A fully software-based audio mixer that mixes PCM audio on the CPU.
///
/// The mixer maintains a set of loaded clips and active channels (voices).
/// Each call to [`update`](AudioMixer::update) renders `buffer_size` frames of
/// audio into the output buffer.  Bus routing with hierarchical volume and
/// priority-based voice stealing are fully implemented.
pub struct SoftwareMixer {
    /// Loaded audio clips.
    clips: Vec<AudioClip>,
    /// Map from clip name to index in `clips`.
    clip_lookup: HashMap<String, usize>,
    /// Currently active voices.
    pub channels: Vec<AudioChannel>,
    /// Bus state map.
    buses: HashMap<AudioBus, BusState>,
    /// Next handle ID (monotonically increasing).
    next_handle_id: u64,
    /// Maximum simultaneous voices.
    max_voices: u32,
    /// Output sample rate.
    pub sample_rate: u32,
    /// Output channels (1 or 2).
    pub output_channels: u16,
    /// Number of frames per update tick.
    pub buffer_size: usize,
    /// Master volume.
    master_volume: f32,
    /// The final mixed output buffer from the last `update` call.
    /// Stereo interleaved, length = buffer_size * output_channels.
    pub output_buffer: Vec<f32>,
}

impl SoftwareMixer {
    /// Create a new software mixer.
    ///
    /// * `sample_rate` - output sample rate (e.g. 44100 or 48000).
    /// * `output_channels` - 1 for mono, 2 for stereo.
    /// * `buffer_size` - number of *frames* per mixing tick.
    /// * `max_voices` - maximum simultaneous playback voices.
    pub fn new(sample_rate: u32, output_channels: u16, buffer_size: usize, max_voices: u32) -> Self {
        let mut buses = HashMap::new();
        for &bus in AudioBus::all_builtin() {
            buses.insert(bus, BusState::new());
        }

        let total_samples = buffer_size * output_channels as usize;

        SoftwareMixer {
            clips: Vec::new(),
            clip_lookup: HashMap::new(),
            channels: Vec::new(),
            buses,
            next_handle_id: 1,
            max_voices,
            sample_rate,
            output_channels,
            buffer_size,
            master_volume: 1.0,
            output_buffer: vec![0.0; total_samples],
        }
    }

    /// Load an audio clip into the mixer's clip registry.
    /// Returns the index of the clip.
    pub fn load_clip(&mut self, clip: AudioClip) -> usize {
        let idx = self.clips.len();
        self.clip_lookup.insert(clip.name.clone(), idx);
        self.clips.push(clip);
        idx
    }

    /// Retrieve a loaded clip by name.
    pub fn get_clip(&self, name: &str) -> Option<&AudioClip> {
        self.clip_lookup.get(name).map(|&i| &self.clips[i])
    }

    /// Number of loaded clips.
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Read the final mixed output buffer (from the most recent `update`).
    pub fn output(&self) -> &[f32] {
        &self.output_buffer
    }

    // -- internal helpers --

    fn alloc_handle(&mut self) -> MixerHandle {
        let h = MixerHandle(self.next_handle_id);
        self.next_handle_id += 1;
        h
    }

    fn find_channel_index(&self, handle: MixerHandle) -> Option<usize> {
        self.channels.iter().position(|c| c.handle == handle)
    }

    /// If the voice limit is reached, try to steal the lowest-priority voice
    /// that has a priority strictly less than `incoming_priority`.
    fn steal_voice_if_needed(&mut self, incoming_priority: u8) -> bool {
        if (self.channels.len() as u32) < self.max_voices {
            return true; // room available
        }
        // Find the channel with the lowest priority.
        let mut worst_idx: Option<usize> = None;
        let mut worst_pri: u8 = u8::MAX;
        for (i, ch) in self.channels.iter().enumerate() {
            if ch.priority < worst_pri {
                worst_pri = ch.priority;
                worst_idx = Some(i);
            }
        }
        if let Some(idx) = worst_idx {
            if worst_pri < incoming_priority {
                log::debug!(
                    "Voice steal: replacing handle {:?} (pri {}) with incoming pri {}",
                    self.channels[idx].handle,
                    worst_pri,
                    incoming_priority,
                );
                self.channels.swap_remove(idx);
                return true;
            }
        }
        false
    }

    /// Ensure bus buffer is the right size and zeroed.
    fn prepare_bus_buffers(&mut self) {
        let total = self.buffer_size * self.output_channels as usize;
        for bus_state in self.buses.values_mut() {
            bus_state.buffer.resize(total, 0.0);
            for s in bus_state.buffer.iter_mut() {
                *s = 0.0;
            }
        }
    }

    /// Process fade envelopes for all channels and remove finished ones.
    fn process_fades(&mut self, dt: f32) {
        for ch in self.channels.iter_mut() {
            if ch.fade_duration <= 0.0 {
                continue;
            }
            ch.fade_elapsed += dt;
            let t = (ch.fade_elapsed / ch.fade_duration).min(1.0);
            ch.effective_volume = ch.fade_start_volume + t * (ch.fade_end_volume - ch.fade_start_volume);

            if t >= 1.0 {
                ch.fade_duration = 0.0;
                ch.fade_elapsed = 0.0;
                match ch.state {
                    ChannelState::FadingIn => {
                        ch.state = ChannelState::Playing;
                        ch.effective_volume = ch.fade_end_volume;
                    }
                    ChannelState::FadingOut | ChannelState::Stopping => {
                        ch.state = ChannelState::Stopped;
                        ch.effective_volume = 0.0;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Mix all active voices into per-bus buffers.
    fn mix_voices(&mut self) {
        let buffer_size = self.buffer_size;
        let output_channels = self.output_channels as usize;
        let mixer_sr = self.sample_rate as f64;

        // We need to iterate over channels mutably while reading clips immutably.
        // Separate the borrows by collecting clip data references into a local slice.
        let clips = &self.clips;

        // Collect bus keys so we can safely borrow later.
        // We'll mix into a temporary vec and then add to bus buffers afterward.
        struct VoiceMixResult {
            bus: AudioBus,
            buffer: Vec<f32>,
        }

        let mut voice_results: Vec<VoiceMixResult> = Vec::with_capacity(self.channels.len());

        for ch in self.channels.iter_mut() {
            if ch.state == ChannelState::Paused || ch.state == ChannelState::Stopped {
                voice_results.push(VoiceMixResult {
                    bus: ch.source.bus,
                    buffer: vec![0.0; buffer_size * output_channels],
                });
                continue;
            }

            let clip = &clips[ch.clip_index];
            let clip_channels = clip.channels as usize;
            let clip_frames = clip.frame_count();
            if clip_frames == 0 {
                voice_results.push(VoiceMixResult {
                    bus: ch.source.bus,
                    buffer: vec![0.0; buffer_size * output_channels],
                });
                continue;
            }

            // Compute playback rate accounting for pitch and sample-rate conversion.
            let rate = ch.source.pitch as f64 * (clip.sample_rate as f64 / mixer_sr);

            let volume = ch.effective_volume;
            let pan = ch.pan.clamp(-1.0, 1.0);
            // Constant-power pan: left = cos(angle), right = sin(angle)
            // where angle = (pan+1)/2 * pi/2
            let angle = ((pan + 1.0) * 0.5) * std::f32::consts::FRAC_PI_2;
            let gain_l = angle.cos() * volume;
            let gain_r = angle.sin() * volume;

            let mut voice_buf = vec![0.0f32; buffer_size * output_channels];
            let mut pos = ch.playback_pos;

            for frame in 0..buffer_size {
                let frame_idx = pos as usize;

                // Check if past end of clip.
                if frame_idx >= clip_frames {
                    if ch.source.looping {
                        pos -= clip_frames as f64;
                        // Continue from the wrapped position (recursion-safe
                        // because rate < clip_frames in any sane scenario).
                        let frame_idx = (pos as usize).min(clip_frames.saturating_sub(1));
                        let frac = (pos - pos.floor()) as f32;
                        let mono_sample = Self::interpolate_frame(clip, frame_idx, clip_channels, clip_frames, frac);
                        Self::write_stereo_frame(&mut voice_buf, frame, output_channels, mono_sample, clip_channels, clip, frame_idx, frac, gain_l, gain_r);
                    } else {
                        // Clip ended -- mark stopped.
                        ch.state = ChannelState::Stopped;
                        break;
                    }
                } else {
                    let frac = (pos - pos.floor()) as f32;
                    Self::write_stereo_frame_from_clip(
                        &mut voice_buf, frame, output_channels,
                        clip, frame_idx, clip_channels, clip_frames,
                        frac, gain_l, gain_r,
                    );
                }

                pos += rate;
            }

            ch.playback_pos = pos;

            voice_results.push(VoiceMixResult {
                bus: ch.source.bus,
                buffer: voice_buf,
            });
        }

        // Accumulate voice results into bus buffers.
        for vr in voice_results {
            if let Some(bus_state) = self.buses.get_mut(&vr.bus) {
                for (i, s) in vr.buffer.iter().enumerate() {
                    if i < bus_state.buffer.len() {
                        bus_state.buffer[i] += s;
                    }
                }
            } else {
                // Route to master if bus not found.
                if let Some(master) = self.buses.get_mut(&AudioBus::Master) {
                    for (i, s) in vr.buffer.iter().enumerate() {
                        if i < master.buffer.len() {
                            master.buffer[i] += s;
                        }
                    }
                }
            }
        }
    }

    /// Linearly interpolate a single channel sample at a fractional frame position.
    fn interpolate_sample(clip: &AudioClip, frame: usize, channel: usize, clip_channels: usize, clip_frames: usize) -> f32 {
        let idx = frame * clip_channels + channel;
        if idx < clip.samples.len() {
            clip.samples[idx]
        } else {
            0.0
        }
    }

    fn interpolate_frame(clip: &AudioClip, frame: usize, clip_channels: usize, clip_frames: usize, frac: f32) -> f32 {
        // For a mono clip return the interpolated sample; for stereo return the average.
        if clip_channels == 1 {
            let s0 = Self::interpolate_sample(clip, frame, 0, clip_channels, clip_frames);
            let s1 = if frame + 1 < clip_frames {
                Self::interpolate_sample(clip, frame + 1, 0, clip_channels, clip_frames)
            } else {
                s0
            };
            s0 + frac * (s1 - s0)
        } else {
            let mut acc = 0.0f32;
            for c in 0..clip_channels {
                let s0 = Self::interpolate_sample(clip, frame, c, clip_channels, clip_frames);
                let s1 = if frame + 1 < clip_frames {
                    Self::interpolate_sample(clip, frame + 1, c, clip_channels, clip_frames)
                } else {
                    s0
                };
                acc += s0 + frac * (s1 - s0);
            }
            acc / clip_channels as f32
        }
    }

    /// Write a stereo output frame from clip data with linear interpolation and panning.
    fn write_stereo_frame(
        buf: &mut [f32],
        frame: usize,
        out_ch: usize,
        mono_sample: f32,
        _clip_channels: usize,
        _clip: &AudioClip,
        _frame_idx: usize,
        _frac: f32,
        gain_l: f32,
        gain_r: f32,
    ) {
        if out_ch >= 2 {
            buf[frame * out_ch] += mono_sample * gain_l;
            buf[frame * out_ch + 1] += mono_sample * gain_r;
        } else {
            buf[frame * out_ch] += mono_sample * (gain_l + gain_r) * 0.5;
        }
    }

    /// Write stereo frame reading directly from clip (handles mono/stereo sources).
    fn write_stereo_frame_from_clip(
        buf: &mut [f32],
        frame: usize,
        out_ch: usize,
        clip: &AudioClip,
        frame_idx: usize,
        clip_channels: usize,
        clip_frames: usize,
        frac: f32,
        gain_l: f32,
        gain_r: f32,
    ) {
        if clip_channels == 1 {
            // Mono source: interpolate single channel, pan to stereo.
            let s0 = Self::interpolate_sample(clip, frame_idx, 0, clip_channels, clip_frames);
            let s1 = if frame_idx + 1 < clip_frames {
                Self::interpolate_sample(clip, frame_idx + 1, 0, clip_channels, clip_frames)
            } else {
                s0
            };
            let sample = s0 + frac * (s1 - s0);
            if out_ch >= 2 {
                buf[frame * out_ch] += sample * gain_l;
                buf[frame * out_ch + 1] += sample * gain_r;
            } else {
                buf[frame * out_ch] += sample * (gain_l + gain_r) * 0.5;
            }
        } else {
            // Stereo source: interpolate L and R independently.
            let l0 = Self::interpolate_sample(clip, frame_idx, 0, clip_channels, clip_frames);
            let l1 = if frame_idx + 1 < clip_frames {
                Self::interpolate_sample(clip, frame_idx + 1, 0, clip_channels, clip_frames)
            } else {
                l0
            };
            let left = l0 + frac * (l1 - l0);

            let r_ch = if clip_channels > 1 { 1 } else { 0 };
            let r0 = Self::interpolate_sample(clip, frame_idx, r_ch, clip_channels, clip_frames);
            let r1 = if frame_idx + 1 < clip_frames {
                Self::interpolate_sample(clip, frame_idx + 1, r_ch, clip_channels, clip_frames)
            } else {
                r0
            };
            let right = r0 + frac * (r1 - r0);

            if out_ch >= 2 {
                buf[frame * out_ch] += left * gain_l;
                buf[frame * out_ch + 1] += right * gain_r;
            } else {
                buf[frame * out_ch] += (left + right) * 0.5 * (gain_l + gain_r) * 0.5;
            }
        }
    }

    /// Sum child buses into the master bus, apply bus volumes.
    fn mix_buses(&mut self) {
        let total = self.buffer_size * self.output_channels as usize;

        // Collect non-master bus data (volume-scaled) and accumulate into master.
        // We must avoid double-borrowing `self.buses`, so collect first.
        let child_buses: Vec<AudioBus> = self
            .buses
            .keys()
            .copied()
            .filter(|b| *b != AudioBus::Master)
            .collect();

        // Apply volume to each child bus and accumulate into a temporary master buffer.
        let mut master_accum = vec![0.0f32; total];

        // First, grab the existing master buffer content (voices routed directly
        // to Master).
        if let Some(ms) = self.buses.get(&AudioBus::Master) {
            let vol = ms.effective_volume();
            for (i, s) in ms.buffer.iter().enumerate() {
                if i < master_accum.len() {
                    master_accum[i] = s * vol;
                }
            }
        }

        for bus in child_buses {
            if let Some(bs) = self.buses.get(&bus) {
                let vol = bs.effective_volume();
                for (i, s) in bs.buffer.iter().enumerate() {
                    if i < master_accum.len() {
                        master_accum[i] += s * vol;
                    }
                }
            }
        }

        // Apply master volume and clamp.
        let master_vol = self.master_volume;
        for s in master_accum.iter_mut() {
            *s *= master_vol;
            *s = s.clamp(-1.0, 1.0);
        }

        // Write to output buffer.
        self.output_buffer.resize(total, 0.0);
        self.output_buffer.copy_from_slice(&master_accum);
    }

    /// Remove stopped channels.
    fn reap_stopped(&mut self) {
        self.channels.retain(|ch| ch.state != ChannelState::Stopped);
    }
}

impl AudioMixer for SoftwareMixer {
    fn play(&mut self, source: AudioSource) -> AudioResult<MixerHandle> {
        // Look up the clip.
        let clip_index = *self
            .clip_lookup
            .get(&source.clip_name)
            .ok_or_else(|| AudioError::UnsupportedFormat(format!("clip not found: {}", source.clip_name)))?;

        // Voice limit check / steal.
        if !self.steal_voice_if_needed(source.priority) {
            return Err(AudioError::VoiceLimitReached(self.max_voices));
        }

        let handle = self.alloc_handle();

        // Compute initial frame from start_offset.
        let clip = &self.clips[clip_index];
        let initial_frame = (source.start_offset * clip.sample_rate as f32) as f64;

        // Ensure the bus exists.
        self.buses.entry(source.bus).or_insert_with(BusState::new);

        let channel = AudioChannel::new(handle, source, clip_index, initial_frame);
        self.channels.push(channel);

        log::debug!("Playing clip '{}' -> handle {:?}", self.clips[clip_index].name, handle);

        Ok(handle)
    }

    fn stop(&mut self, handle: MixerHandle, fade_out: f32) -> AudioResult<()> {
        let idx = self
            .find_channel_index(handle)
            .ok_or(AudioError::InvalidHandle(handle))?;
        let ch = &mut self.channels[idx];
        if fade_out > 0.0 {
            ch.state = ChannelState::FadingOut;
            ch.fade_elapsed = 0.0;
            ch.fade_duration = fade_out;
            ch.fade_start_volume = ch.effective_volume;
            ch.fade_end_volume = 0.0;
        } else {
            ch.state = ChannelState::Stopped;
        }
        Ok(())
    }

    fn pause(&mut self, handle: MixerHandle) -> AudioResult<()> {
        let idx = self
            .find_channel_index(handle)
            .ok_or(AudioError::InvalidHandle(handle))?;
        if self.channels[idx].state == ChannelState::Playing
            || self.channels[idx].state == ChannelState::FadingIn
        {
            self.channels[idx].state = ChannelState::Paused;
        }
        Ok(())
    }

    fn resume(&mut self, handle: MixerHandle) -> AudioResult<()> {
        let idx = self
            .find_channel_index(handle)
            .ok_or(AudioError::InvalidHandle(handle))?;
        if self.channels[idx].state == ChannelState::Paused {
            self.channels[idx].state = ChannelState::Playing;
        }
        Ok(())
    }

    fn set_volume(&mut self, handle: MixerHandle, volume: f32) -> AudioResult<()> {
        let idx = self
            .find_channel_index(handle)
            .ok_or(AudioError::InvalidHandle(handle))?;
        self.channels[idx].source.volume = volume;
        // If not currently fading, update effective volume immediately.
        if self.channels[idx].fade_duration <= 0.0 {
            self.channels[idx].effective_volume = volume;
        }
        Ok(())
    }

    fn set_pitch(&mut self, handle: MixerHandle, pitch: f32) -> AudioResult<()> {
        let idx = self
            .find_channel_index(handle)
            .ok_or(AudioError::InvalidHandle(handle))?;
        self.channels[idx].source.pitch = pitch.max(0.01);
        Ok(())
    }

    fn set_bus_volume(&mut self, bus: AudioBus, volume: f32) {
        self.buses.entry(bus).or_insert_with(BusState::new).volume = volume.max(0.0);
    }

    fn set_bus_mute(&mut self, bus: AudioBus, muted: bool) {
        self.buses.entry(bus).or_insert_with(BusState::new).muted = muted;
    }

    fn channel_state(&self, handle: MixerHandle) -> AudioResult<ChannelState> {
        self.find_channel_index(handle)
            .map(|i| self.channels[i].state)
            .ok_or(AudioError::InvalidHandle(handle))
    }

    fn stop_all(&mut self) {
        for ch in &mut self.channels {
            ch.state = ChannelState::Stopped;
        }
    }

    fn pause_all(&mut self) {
        for ch in &mut self.channels {
            if ch.state == ChannelState::Playing || ch.state == ChannelState::FadingIn {
                ch.state = ChannelState::Paused;
            }
        }
    }

    fn resume_all(&mut self) {
        for ch in &mut self.channels {
            if ch.state == ChannelState::Paused {
                ch.state = ChannelState::Playing;
            }
        }
    }

    fn update(&mut self, dt: f32) {
        profiling::scope!("SoftwareMixer::update");

        // 1. Process fades.
        self.process_fades(dt);

        // 2. Prepare bus buffers.
        self.prepare_bus_buffers();

        // 3. Mix all voices into bus buffers.
        self.mix_voices();

        // 4. Sum buses into master output.
        self.mix_buses();

        // 5. Remove stopped channels.
        self.reap_stopped();
    }

    fn max_voices(&self) -> u32 {
        self.max_voices
    }

    fn active_voice_count(&self) -> u32 {
        self.channels
            .iter()
            .filter(|c| c.state != ChannelState::Stopped)
            .count() as u32
    }

    fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = volume.max(0.0);
    }

    fn master_volume(&self) -> f32 {
        self.master_volume
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid WAV file (mono, 16-bit, 44100 Hz) from sample data.
    fn build_wav_i16(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
        let bits_per_sample: u16 = 16;
        let byte_rate = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
        let block_align = channels * (bits_per_sample / 8);
        let data_size = (samples.len() * 2) as u32;
        let file_size = 36 + data_size;

        let mut buf = Vec::with_capacity(file_size as usize + 8);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");

        // fmt chunk
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }

        buf
    }

    fn build_wav_f32(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
        let bits_per_sample: u16 = 32;
        let byte_rate = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
        let block_align = channels * (bits_per_sample / 8);
        let data_size = (samples.len() * 4) as u32;
        let file_size = 36 + data_size;

        let mut buf = Vec::with_capacity(file_size as usize + 8);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");

        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());

        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }

        buf
    }

    #[test]
    fn wav_parse_i16_mono() {
        let raw: Vec<i16> = vec![0, 16384, 32767, -32768, -16384, 0];
        let wav = build_wav_i16(&raw, 44100, 1);
        let clip = AudioClip::from_wav_bytes(&wav).unwrap();
        assert_eq!(clip.sample_rate, 44100);
        assert_eq!(clip.channels, 1);
        assert_eq!(clip.samples.len(), 6);
        // 0 maps to 0.0
        assert!((clip.samples[0] - 0.0).abs() < 1e-4);
        // 16384 maps to ~0.5
        assert!((clip.samples[1] - 0.5).abs() < 0.01);
        // 32767 maps to ~1.0
        assert!((clip.samples[2] - 1.0).abs() < 0.001);
        // -32768 maps to -1.0
        assert!((clip.samples[3] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn wav_parse_f32_stereo() {
        let raw: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0];
        let wav = build_wav_f32(&raw, 48000, 2);
        let clip = AudioClip::from_wav_bytes(&wav).unwrap();
        assert_eq!(clip.sample_rate, 48000);
        assert_eq!(clip.channels, 2);
        assert_eq!(clip.samples.len(), 4);
        assert!((clip.samples[0] - 0.5).abs() < 1e-6);
        assert!((clip.samples[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn wav_parse_rejects_too_small() {
        let tiny = vec![0u8; 10];
        assert!(AudioClip::from_wav_bytes(&tiny).is_err());
    }

    #[test]
    fn wav_parse_rejects_bad_magic() {
        let mut wav = build_wav_i16(&[0i16; 10], 44100, 1);
        wav[0] = b'X'; // corrupt RIFF
        assert!(AudioClip::from_wav_bytes(&wav).is_err());
    }

    #[test]
    fn clip_duration() {
        let clip = AudioClip::sine_wave(440.0, 1.0, 44100);
        assert!((clip.duration() - 1.0).abs() < 0.001);
    }

    #[test]
    fn clip_silence() {
        let clip = AudioClip::silence(0.5, 44100, 2);
        assert_eq!(clip.channels, 2);
        assert_eq!(clip.frame_count(), 22050);
        assert!(clip.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn mixer_play_and_update() {
        let mut mixer = SoftwareMixer::new(44100, 2, 512, 32);
        let clip = AudioClip::sine_wave(440.0, 0.5, 44100);
        mixer.load_clip(clip);

        let handle = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                volume: 0.8,
                ..Default::default()
            })
            .unwrap();

        assert_eq!(mixer.active_voice_count(), 1);
        assert_eq!(mixer.channel_state(handle).unwrap(), ChannelState::Playing);

        // Run one update tick.
        mixer.update(1.0 / 60.0);

        // Output buffer should be non-silent.
        let non_zero = mixer.output().iter().any(|&s| s != 0.0);
        assert!(non_zero, "output should contain non-zero samples after playing a sine wave");
    }

    #[test]
    fn mixer_pause_resume() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 16);
        mixer.load_clip(AudioClip::sine_wave(220.0, 1.0, 44100));

        let h = mixer
            .play(AudioSource {
                clip_name: "sine_220hz".into(),
                ..Default::default()
            })
            .unwrap();

        mixer.pause(h).unwrap();
        assert_eq!(mixer.channel_state(h).unwrap(), ChannelState::Paused);

        mixer.resume(h).unwrap();
        assert_eq!(mixer.channel_state(h).unwrap(), ChannelState::Playing);
    }

    #[test]
    fn mixer_stop_immediate() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 16);
        mixer.load_clip(AudioClip::sine_wave(330.0, 1.0, 44100));

        let h = mixer
            .play(AudioSource {
                clip_name: "sine_330hz".into(),
                ..Default::default()
            })
            .unwrap();

        mixer.stop(h, 0.0).unwrap();
        assert_eq!(mixer.channel_state(h).unwrap(), ChannelState::Stopped);

        // After update, the stopped channel is reaped.
        mixer.update(1.0 / 60.0);
        assert_eq!(mixer.active_voice_count(), 0);
    }

    #[test]
    fn mixer_stop_with_fade() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 16);
        mixer.load_clip(AudioClip::sine_wave(440.0, 2.0, 44100));

        let h = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                ..Default::default()
            })
            .unwrap();

        mixer.stop(h, 0.5).unwrap();
        assert_eq!(mixer.channel_state(h).unwrap(), ChannelState::FadingOut);

        // After enough updates to exceed the fade, it should be stopped.
        for _ in 0..40 {
            mixer.update(1.0 / 60.0);
        }
        // Channel should have been reaped.
        assert_eq!(mixer.active_voice_count(), 0);
    }

    #[test]
    fn mixer_voice_stealing() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 2); // max 2 voices
        mixer.load_clip(AudioClip::sine_wave(440.0, 1.0, 44100));

        // Fill all voices with low priority.
        let _h1 = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                priority: 10,
                ..Default::default()
            })
            .unwrap();
        let _h2 = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                priority: 20,
                ..Default::default()
            })
            .unwrap();

        assert_eq!(mixer.active_voice_count(), 2);

        // A higher-priority voice should steal the lowest.
        let h3 = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                priority: 50,
                ..Default::default()
            })
            .unwrap();

        assert_eq!(mixer.active_voice_count(), 2);
        assert!(mixer.channel_state(h3).is_ok());
    }

    #[test]
    fn mixer_voice_limit_rejected() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 1);
        mixer.load_clip(AudioClip::sine_wave(440.0, 1.0, 44100));

        let _h1 = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                priority: 200,
                ..Default::default()
            })
            .unwrap();

        // Trying to play with a *lower* priority should fail.
        let result = mixer.play(AudioSource {
            clip_name: "sine_440hz".into(),
            priority: 100,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn mixer_bus_volume() {
        let mut mixer = SoftwareMixer::new(44100, 2, 512, 16);
        mixer.load_clip(AudioClip::sine_wave(440.0, 1.0, 44100));

        mixer.set_bus_volume(AudioBus::SFX, 0.5);

        let _h = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                bus: AudioBus::SFX,
                ..Default::default()
            })
            .unwrap();

        mixer.update(1.0 / 60.0);

        // The output should be quieter than full volume.
        let peak: f32 = mixer.output().iter().map(|s| s.abs()).fold(0.0, f32::max);
        assert!(
            peak < 0.8,
            "peak {peak} should be reduced by bus volume 0.5"
        );
    }

    #[test]
    fn mixer_bus_mute() {
        let mut mixer = SoftwareMixer::new(44100, 2, 512, 16);
        mixer.load_clip(AudioClip::sine_wave(440.0, 1.0, 44100));
        mixer.set_bus_mute(AudioBus::SFX, true);

        let _h = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                bus: AudioBus::SFX,
                ..Default::default()
            })
            .unwrap();

        mixer.update(1.0 / 60.0);

        let peak: f32 = mixer.output().iter().map(|s| s.abs()).fold(0.0, f32::max);
        assert!(peak < 1e-6, "muted bus should produce silence, got peak {peak}");
    }

    #[test]
    fn mixer_master_volume() {
        let mut mixer = SoftwareMixer::new(44100, 2, 512, 16);
        mixer.load_clip(AudioClip::sine_wave(440.0, 1.0, 44100));
        mixer.set_master_volume(0.0);

        let _h = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                ..Default::default()
            })
            .unwrap();

        mixer.update(1.0 / 60.0);
        let peak: f32 = mixer.output().iter().map(|s| s.abs()).fold(0.0, f32::max);
        assert!(peak < 1e-6, "master volume 0 should produce silence");
    }

    #[test]
    fn mixer_set_volume_and_pitch() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 16);
        mixer.load_clip(AudioClip::sine_wave(440.0, 1.0, 44100));

        let h = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                ..Default::default()
            })
            .unwrap();

        mixer.set_volume(h, 0.3).unwrap();
        mixer.set_pitch(h, 2.0).unwrap();

        mixer.update(1.0 / 60.0);
        // Just verify it doesn't panic and produces output.
        let non_zero = mixer.output().iter().any(|&s| s != 0.0);
        assert!(non_zero);
    }

    #[test]
    fn mixer_looping_clip() {
        let mut mixer = SoftwareMixer::new(44100, 2, 44100, 16); // 1 second buffer
        // Very short clip (100 frames at 44100 Hz ≈ 2.3ms).
        let mut clip = AudioClip::sine_wave(440.0, 0.01, 44100);
        clip.name = "short".into();
        mixer.load_clip(clip);

        let h = mixer
            .play(AudioSource {
                clip_name: "short".into(),
                looping: true,
                ..Default::default()
            })
            .unwrap();

        // Run a full second of mixing -- should not stop because looping.
        mixer.update(1.0);
        assert_eq!(mixer.channel_state(h).unwrap(), ChannelState::Playing);
    }

    #[test]
    fn mixer_non_looping_clip_stops() {
        let mut mixer = SoftwareMixer::new(44100, 2, 44100, 16);
        let mut clip = AudioClip::sine_wave(440.0, 0.01, 44100);
        clip.name = "short".into();
        mixer.load_clip(clip);

        let h = mixer
            .play(AudioSource {
                clip_name: "short".into(),
                looping: false,
                ..Default::default()
            })
            .unwrap();

        mixer.update(1.0);
        // Channel should have been reaped after the clip ended.
        assert!(mixer.channel_state(h).is_err());
        assert_eq!(mixer.active_voice_count(), 0);
    }

    #[test]
    fn mixer_stop_all() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 16);
        mixer.load_clip(AudioClip::sine_wave(440.0, 1.0, 44100));

        for _ in 0..5 {
            mixer
                .play(AudioSource {
                    clip_name: "sine_440hz".into(),
                    ..Default::default()
                })
                .unwrap();
        }
        assert_eq!(mixer.active_voice_count(), 5);

        mixer.stop_all();
        mixer.update(1.0 / 60.0); // reap
        assert_eq!(mixer.active_voice_count(), 0);
    }

    #[test]
    fn mixer_pause_all_resume_all() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 16);
        mixer.load_clip(AudioClip::sine_wave(440.0, 1.0, 44100));

        let h1 = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                ..Default::default()
            })
            .unwrap();
        let h2 = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                ..Default::default()
            })
            .unwrap();

        mixer.pause_all();
        assert_eq!(mixer.channel_state(h1).unwrap(), ChannelState::Paused);
        assert_eq!(mixer.channel_state(h2).unwrap(), ChannelState::Paused);

        mixer.resume_all();
        assert_eq!(mixer.channel_state(h1).unwrap(), ChannelState::Playing);
        assert_eq!(mixer.channel_state(h2).unwrap(), ChannelState::Playing);
    }

    #[test]
    fn mixer_output_clamped() {
        let mut mixer = SoftwareMixer::new(44100, 2, 512, 64);
        // Load a loud clip.
        let mut clip = AudioClip::sine_wave(440.0, 1.0, 44100);
        clip.name = "loud".into();
        mixer.load_clip(clip);

        // Play many instances at full volume to exceed 1.0.
        for _ in 0..20 {
            mixer
                .play(AudioSource {
                    clip_name: "loud".into(),
                    volume: 1.0,
                    ..Default::default()
                })
                .unwrap();
        }

        mixer.update(1.0 / 60.0);

        // Verify clamping.
        for &s in mixer.output() {
            assert!(s >= -1.0 && s <= 1.0, "sample {s} out of [-1,1]");
        }
    }

    #[test]
    fn mixer_fade_in() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 16);
        mixer.load_clip(AudioClip::sine_wave(440.0, 2.0, 44100));

        let h = mixer
            .play(AudioSource {
                clip_name: "sine_440hz".into(),
                fade_in: 0.5,
                ..Default::default()
            })
            .unwrap();

        assert_eq!(mixer.channel_state(h).unwrap(), ChannelState::FadingIn);

        // After 0.5s of updates it should transition to Playing.
        for _ in 0..35 {
            mixer.update(1.0 / 60.0);
        }
        assert_eq!(mixer.channel_state(h).unwrap(), ChannelState::Playing);
    }

    #[test]
    fn wav_parse_u8() {
        // Build a WAV with 8-bit unsigned samples.
        let samples_u8: Vec<u8> = vec![0, 128, 255]; // min, mid, max
        let sample_rate: u32 = 22050;
        let channels: u16 = 1;
        let bits_per_sample: u16 = 8;
        let byte_rate = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
        let block_align = channels * (bits_per_sample / 8);
        let data_size = samples_u8.len() as u32;
        let file_size = 36 + data_size;

        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits_per_sample.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        wav.extend_from_slice(&samples_u8);

        let clip = AudioClip::from_wav_bytes(&wav).unwrap();
        assert_eq!(clip.channels, 1);
        assert_eq!(clip.sample_rate, 22050);
        // 0 -> -1.0, 128 -> 0.0, 255 -> ~1.0
        assert!((clip.samples[0] - (-1.0)).abs() < 0.01);
        assert!((clip.samples[1] - 0.0).abs() < 0.01);
        assert!((clip.samples[2] - 1.0).abs() < 0.02);
    }

    #[test]
    fn bus_parent_routing() {
        assert_eq!(AudioBus::Master.parent(), None);
        assert_eq!(AudioBus::SFX.parent(), Some(AudioBus::Master));
        assert_eq!(AudioBus::Music.parent(), Some(AudioBus::Master));
        assert_eq!(AudioBus::Voice.parent(), Some(AudioBus::Master));
        assert_eq!(AudioBus::Ambient.parent(), Some(AudioBus::Master));
        assert_eq!(AudioBus::Custom(5).parent(), Some(AudioBus::Master));
    }

    #[test]
    fn clip_from_wav_named() {
        let raw: Vec<i16> = vec![0; 100];
        let wav = build_wav_i16(&raw, 44100, 1);
        let clip = AudioClip::from_wav_bytes_named(&wav, "test_clip").unwrap();
        assert_eq!(clip.name, "test_clip");
    }

    #[test]
    fn mixer_load_clip() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 16);
        let clip = AudioClip::sine_wave(440.0, 0.5, 44100);
        mixer.load_clip(clip);
        assert_eq!(mixer.clip_count(), 1);
        assert!(mixer.get_clip("sine_440hz").is_some());
    }

    #[test]
    fn mixer_play_unknown_clip_fails() {
        let mut mixer = SoftwareMixer::new(44100, 2, 256, 16);
        let result = mixer.play(AudioSource {
            clip_name: "nonexistent".into(),
            ..Default::default()
        });
        assert!(result.is_err());
    }
}
