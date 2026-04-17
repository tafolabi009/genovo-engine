// engine/audio/src/audio_streaming.rs
//
// Streaming audio playback for the Genovo engine.
//
// Provides chunk-based audio streaming for large audio files:
//
// - **Chunk-based reading** -- Read audio data in configurable chunks from disk.
// - **Double-buffered streaming** -- Two buffers alternate to ensure seamless
//   playback while the next chunk loads.
// - **Seek support** -- Seek to any sample position in the stream.
// - **Loop points** -- Define start/end loop regions within the audio.
// - **Streaming from asset system** -- Integration with the engine asset pipeline.
// - **Memory-limited playback** -- Play large files with fixed memory footprint.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default chunk size in samples (per channel).
const DEFAULT_CHUNK_SAMPLES: usize = 44100; // ~1 second at 44.1kHz

/// Default number of buffers for double-buffering.
const DEFAULT_BUFFER_COUNT: usize = 2;

/// Maximum simultaneous streams.
const MAX_STREAMS: usize = 32;

// ---------------------------------------------------------------------------
// Audio format
// ---------------------------------------------------------------------------

/// Audio format for streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamAudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
}

impl StreamAudioFormat {
    pub fn new(sample_rate: u32, channels: u16, bits_per_sample: u16) -> Self {
        Self { sample_rate, channels, bits_per_sample }
    }

    /// Standard CD quality: 44100 Hz, stereo, 16-bit.
    pub fn cd_quality() -> Self {
        Self::new(44100, 2, 16)
    }

    /// Bytes per sample (all channels).
    pub fn bytes_per_sample(&self) -> u32 {
        self.channels as u32 * self.bits_per_sample as u32 / 8
    }

    /// Bytes per second.
    pub fn bytes_per_second(&self) -> u32 {
        self.sample_rate * self.bytes_per_sample()
    }

    /// Convert sample count to duration in seconds.
    pub fn samples_to_seconds(&self, samples: u64) -> f64 {
        samples as f64 / self.sample_rate as f64
    }

    /// Convert seconds to sample count.
    pub fn seconds_to_samples(&self, seconds: f64) -> u64 {
        (seconds * self.sample_rate as f64) as u64
    }
}

impl Default for StreamAudioFormat {
    fn default() -> Self {
        Self::cd_quality()
    }
}

// ---------------------------------------------------------------------------
// Loop region
// ---------------------------------------------------------------------------

/// A loop region within an audio stream.
#[derive(Debug, Clone, Copy)]
pub struct LoopRegion {
    /// Start sample of the loop.
    pub start_sample: u64,
    /// End sample of the loop (exclusive).
    pub end_sample: u64,
    /// Number of times to loop (0 = infinite).
    pub loop_count: u32,
    /// Crossfade duration in samples (for smooth looping).
    pub crossfade_samples: u32,
}

impl LoopRegion {
    /// Create a loop from start to end.
    pub fn new(start: u64, end: u64) -> Self {
        Self {
            start_sample: start,
            end_sample: end,
            loop_count: 0,
            crossfade_samples: 0,
        }
    }

    /// Loop the entire file.
    pub fn full(total_samples: u64) -> Self {
        Self::new(0, total_samples)
    }

    /// Duration in samples.
    pub fn duration_samples(&self) -> u64 {
        self.end_sample.saturating_sub(self.start_sample)
    }

    /// Check if a sample position is within the loop.
    pub fn contains(&self, sample: u64) -> bool {
        sample >= self.start_sample && sample < self.end_sample
    }
}

// ---------------------------------------------------------------------------
// Stream buffer
// ---------------------------------------------------------------------------

/// A buffer of audio samples.
#[derive(Debug, Clone)]
pub struct StreamBuffer {
    /// Interleaved f32 sample data.
    pub samples: Vec<f32>,
    /// Starting sample position in the stream.
    pub start_sample: u64,
    /// Number of valid samples in this buffer.
    pub valid_samples: usize,
    /// Whether this buffer has been filled with data.
    pub filled: bool,
    /// Buffer index (for identification).
    pub index: usize,
}

impl StreamBuffer {
    pub fn new(capacity: usize, index: usize) -> Self {
        Self {
            samples: vec![0.0; capacity],
            start_sample: 0,
            valid_samples: 0,
            filled: false,
            index,
        }
    }

    pub fn clear(&mut self) {
        self.samples.fill(0.0);
        self.valid_samples = 0;
        self.filled = false;
    }

    pub fn is_empty(&self) -> bool {
        self.valid_samples == 0
    }

    /// Get a sample at a specific offset within this buffer.
    pub fn get_sample(&self, offset: usize) -> Option<f32> {
        if offset < self.valid_samples {
            Some(self.samples[offset])
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Stream state
// ---------------------------------------------------------------------------

/// Playback state of a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Not yet started.
    Idle,
    /// Loading initial data.
    Buffering,
    /// Actively playing.
    Playing,
    /// Paused (can resume).
    Paused,
    /// Seeking to a new position.
    Seeking,
    /// Reached end of stream.
    Finished,
    /// Error occurred.
    Error,
}

impl Default for StreamState {
    fn default() -> Self {
        Self::Idle
    }
}

// ---------------------------------------------------------------------------
// Stream handle
// ---------------------------------------------------------------------------

/// Handle to an active stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamHandle(pub u32);

impl fmt::Display for StreamHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stream({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Audio data source trait
// ---------------------------------------------------------------------------

/// Trait for audio data sources that provide samples for streaming.
pub trait AudioDataSource: fmt::Debug + Send {
    /// Get the audio format.
    fn format(&self) -> StreamAudioFormat;

    /// Get the total number of samples (per channel).
    fn total_samples(&self) -> u64;

    /// Read samples starting at the given position.
    /// Returns the number of samples actually read.
    fn read_samples(&mut self, position: u64, output: &mut [f32]) -> usize;

    /// Seek to a sample position.
    fn seek(&mut self, position: u64) -> bool;

    /// Get the source name/path.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Memory data source (for testing)
// ---------------------------------------------------------------------------

/// In-memory audio data source.
#[derive(Debug)]
pub struct MemoryDataSource {
    samples: Vec<f32>,
    format: StreamAudioFormat,
    name: String,
}

impl MemoryDataSource {
    pub fn new(samples: Vec<f32>, format: StreamAudioFormat, name: &str) -> Self {
        Self {
            samples,
            format,
            name: name.to_string(),
        }
    }

    /// Generate a sine wave source for testing.
    pub fn sine_wave(frequency: f32, duration: f32, format: StreamAudioFormat) -> Self {
        let total = (format.sample_rate as f32 * duration) as usize * format.channels as usize;
        let mut samples = vec![0.0; total];
        let channels = format.channels as usize;

        for i in 0..(total / channels) {
            let t = i as f32 / format.sample_rate as f32;
            let value = (t * frequency * 2.0 * std::f32::consts::PI).sin() * 0.5;
            for ch in 0..channels {
                samples[i * channels + ch] = value;
            }
        }

        Self::new(samples, format, "sine_wave")
    }
}

impl AudioDataSource for MemoryDataSource {
    fn format(&self) -> StreamAudioFormat {
        self.format
    }

    fn total_samples(&self) -> u64 {
        (self.samples.len() / self.format.channels as usize) as u64
    }

    fn read_samples(&mut self, position: u64, output: &mut [f32]) -> usize {
        let channels = self.format.channels as usize;
        let start = position as usize * channels;
        if start >= self.samples.len() {
            return 0;
        }
        let available = self.samples.len() - start;
        let to_read = output.len().min(available);
        output[..to_read].copy_from_slice(&self.samples[start..start + to_read]);
        to_read / channels
    }

    fn seek(&mut self, _position: u64) -> bool {
        true
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Audio stream
// ---------------------------------------------------------------------------

/// Configuration for an audio stream.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Chunk size in samples per channel.
    pub chunk_samples: usize,
    /// Number of buffers.
    pub buffer_count: usize,
    /// Volume (0..1).
    pub volume: f32,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Loop region (None = no looping).
    pub loop_region: Option<LoopRegion>,
    /// Whether to start playing immediately.
    pub auto_play: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_samples: DEFAULT_CHUNK_SAMPLES,
            buffer_count: DEFAULT_BUFFER_COUNT,
            volume: 1.0,
            speed: 1.0,
            loop_region: None,
            auto_play: true,
        }
    }
}

/// An active audio stream.
pub struct AudioStream {
    handle: StreamHandle,
    source: Box<dyn AudioDataSource>,
    config: StreamConfig,
    format: StreamAudioFormat,
    buffers: Vec<StreamBuffer>,
    state: StreamState,
    /// Current playback position in samples.
    position: u64,
    /// Total samples in the source.
    total_samples: u64,
    /// Current buffer being read from.
    read_buffer: usize,
    /// Current offset within the read buffer.
    read_offset: usize,
    /// Volume (with fade).
    current_volume: f32,
    /// Number of loops completed.
    loops_completed: u32,
    /// Time elapsed.
    elapsed: f64,
    /// Error message (if state is Error).
    error: Option<String>,
}

impl AudioStream {
    /// Create a new stream.
    pub fn new(
        handle: StreamHandle,
        source: Box<dyn AudioDataSource>,
        config: StreamConfig,
    ) -> Self {
        let format = source.format();
        let total_samples = source.total_samples();
        let channels = format.channels as usize;
        let chunk_size = config.chunk_samples * channels;
        let buffer_count = config.buffer_count;

        let buffers: Vec<StreamBuffer> = (0..buffer_count)
            .map(|i| StreamBuffer::new(chunk_size, i))
            .collect();

        let initial_state = if config.auto_play {
            StreamState::Buffering
        } else {
            StreamState::Idle
        };

        Self {
            handle,
            source,
            config,
            format,
            buffers,
            state: initial_state,
            position: 0,
            total_samples,
            read_buffer: 0,
            read_offset: 0,
            current_volume: 1.0,
            loops_completed: 0,
            elapsed: 0.0,
            error: None,
        }
    }

    /// Get the stream handle.
    pub fn handle(&self) -> StreamHandle {
        self.handle
    }

    /// Get the current state.
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Get the playback position in seconds.
    pub fn position_seconds(&self) -> f64 {
        self.format.samples_to_seconds(self.position)
    }

    /// Get the total duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.format.samples_to_seconds(self.total_samples)
    }

    /// Get the playback progress (0..1).
    pub fn progress(&self) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        self.position as f32 / self.total_samples as f32
    }

    /// Play.
    pub fn play(&mut self) {
        if self.state == StreamState::Paused || self.state == StreamState::Idle {
            self.state = StreamState::Playing;
        }
    }

    /// Pause.
    pub fn pause(&mut self) {
        if self.state == StreamState::Playing {
            self.state = StreamState::Paused;
        }
    }

    /// Stop and reset.
    pub fn stop(&mut self) {
        self.state = StreamState::Idle;
        self.position = 0;
        self.read_buffer = 0;
        self.read_offset = 0;
        self.loops_completed = 0;
        for buf in &mut self.buffers {
            buf.clear();
        }
    }

    /// Seek to a position in seconds.
    pub fn seek(&mut self, seconds: f64) {
        let sample = self.format.seconds_to_samples(seconds);
        self.seek_to_sample(sample);
    }

    /// Seek to a sample position.
    pub fn seek_to_sample(&mut self, sample: u64) {
        let clamped = sample.min(self.total_samples);
        self.position = clamped;
        self.source.seek(clamped);
        self.read_offset = 0;
        for buf in &mut self.buffers {
            buf.clear();
        }
        // Refill buffers.
        self.fill_buffers();
    }

    /// Set volume.
    pub fn set_volume(&mut self, volume: f32) {
        self.config.volume = volume.clamp(0.0, 2.0);
    }

    /// Set playback speed.
    pub fn set_speed(&mut self, speed: f32) {
        self.config.speed = speed.clamp(0.1, 4.0);
    }

    /// Set loop region.
    pub fn set_loop(&mut self, region: Option<LoopRegion>) {
        self.config.loop_region = region;
    }

    /// Fill buffers with data from the source.
    fn fill_buffers(&mut self) {
        for buf in &mut self.buffers {
            if !buf.filled {
                let channels = self.format.channels as usize;
                let samples_read = self.source.read_samples(
                    self.position + buf.start_sample,
                    &mut buf.samples,
                );
                buf.valid_samples = samples_read * channels;
                buf.filled = true;
            }
        }
    }

    /// Read samples into an output buffer for mixing.
    pub fn read(&mut self, output: &mut [f32], dt: f32) -> usize {
        if self.state != StreamState::Playing && self.state != StreamState::Buffering {
            return 0;
        }

        if self.state == StreamState::Buffering {
            self.fill_buffers();
            self.state = StreamState::Playing;
        }

        self.elapsed += dt as f64;
        let channels = self.format.channels as usize;
        let mut written = 0;

        while written < output.len() {
            // Check if we need to refill current buffer.
            if self.read_buffer < self.buffers.len() {
                let buf = &self.buffers[self.read_buffer];
                if self.read_offset < buf.valid_samples {
                    let available = buf.valid_samples - self.read_offset;
                    let to_copy = available.min(output.len() - written);
                    for i in 0..to_copy {
                        output[written + i] = buf.samples[self.read_offset + i] * self.config.volume;
                    }
                    self.read_offset += to_copy;
                    written += to_copy;
                    self.position += (to_copy / channels) as u64;
                    continue;
                }
            }

            // Current buffer exhausted.
            // Check for loop.
            if let Some(loop_region) = &self.config.loop_region {
                if self.position >= loop_region.end_sample {
                    let max_loops = loop_region.loop_count;
                    if max_loops == 0 || self.loops_completed < max_loops {
                        self.loops_completed += 1;
                        self.seek_to_sample(loop_region.start_sample);
                        continue;
                    }
                }
            }

            // Check for end of stream.
            if self.position >= self.total_samples {
                self.state = StreamState::Finished;
                break;
            }

            // Advance to next buffer and refill.
            self.read_buffer = (self.read_buffer + 1) % self.buffers.len();
            self.read_offset = 0;

            let buf = &mut self.buffers[self.read_buffer];
            buf.start_sample = self.position;
            buf.filled = false;
            let samples_read = self.source.read_samples(self.position, &mut buf.samples);
            buf.valid_samples = samples_read * channels;
            buf.filled = true;

            if samples_read == 0 {
                self.state = StreamState::Finished;
                break;
            }
        }

        written
    }

    /// Get format info.
    pub fn format(&self) -> &StreamAudioFormat {
        &self.format
    }

    /// Get the source name.
    pub fn source_name(&self) -> &str {
        self.source.name()
    }

    /// Get memory usage of buffers.
    pub fn buffer_memory(&self) -> usize {
        self.buffers.iter().map(|b| b.samples.len() * 4).sum()
    }
}

// ---------------------------------------------------------------------------
// Stream manager
// ---------------------------------------------------------------------------

/// Manages multiple audio streams.
pub struct AudioStreamManager {
    streams: HashMap<StreamHandle, AudioStream>,
    next_handle: u32,
}

impl AudioStreamManager {
    pub fn new() -> Self {
        Self {
            streams: HashMap::new(),
            next_handle: 0,
        }
    }

    /// Create a new stream.
    pub fn create(
        &mut self,
        source: Box<dyn AudioDataSource>,
        config: StreamConfig,
    ) -> StreamHandle {
        let handle = StreamHandle(self.next_handle);
        self.next_handle += 1;
        let stream = AudioStream::new(handle, source, config);
        self.streams.insert(handle, stream);
        handle
    }

    /// Remove a stream.
    pub fn remove(&mut self, handle: StreamHandle) {
        self.streams.remove(&handle);
    }

    /// Get a stream.
    pub fn stream(&self, handle: StreamHandle) -> Option<&AudioStream> {
        self.streams.get(&handle)
    }

    /// Get a mutable stream.
    pub fn stream_mut(&mut self, handle: StreamHandle) -> Option<&mut AudioStream> {
        self.streams.get_mut(&handle)
    }

    /// Update all streams (mix into output).
    pub fn mix_all(&mut self, output: &mut [f32], dt: f32) {
        output.fill(0.0);
        let handles: Vec<StreamHandle> = self.streams.keys().copied().collect();
        let mut temp = vec![0.0_f32; output.len()];

        for handle in handles {
            if let Some(stream) = self.streams.get_mut(&handle) {
                temp.fill(0.0);
                let read = stream.read(&mut temp, dt);
                for i in 0..read.min(output.len()) {
                    output[i] += temp[i];
                }
            }
        }

        // Remove finished streams.
        self.streams.retain(|_, s| s.state() != StreamState::Finished);
    }

    /// Get the number of active streams.
    pub fn active_count(&self) -> usize {
        self.streams.len()
    }

    /// Total buffer memory across all streams.
    pub fn total_buffer_memory(&self) -> usize {
        self.streams.values().map(|s| s.buffer_memory()).sum()
    }
}

impl Default for AudioStreamManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_source() {
        let source = MemoryDataSource::sine_wave(440.0, 1.0, StreamAudioFormat::cd_quality());
        assert!(source.total_samples() > 0);
    }

    #[test]
    fn test_stream_playback() {
        let source = MemoryDataSource::sine_wave(440.0, 0.1, StreamAudioFormat::new(44100, 1, 16));
        let config = StreamConfig { auto_play: true, ..Default::default() };
        let mut stream = AudioStream::new(StreamHandle(0), Box::new(source), config);

        let mut output = vec![0.0_f32; 4410];
        let read = stream.read(&mut output, 0.1);
        assert!(read > 0);
        assert!(output.iter().any(|&s| s != 0.0));
    }

    #[test]
    fn test_seek() {
        let source = MemoryDataSource::sine_wave(440.0, 2.0, StreamAudioFormat::new(44100, 1, 16));
        let mut stream = AudioStream::new(StreamHandle(0), Box::new(source), StreamConfig::default());
        stream.seek(1.0);
        assert!((stream.position_seconds() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_loop() {
        let source = MemoryDataSource::sine_wave(440.0, 0.05, StreamAudioFormat::new(44100, 1, 16));
        let total = source.total_samples();
        let config = StreamConfig {
            auto_play: true,
            loop_region: Some(LoopRegion::full(total)),
            ..Default::default()
        };
        let mut stream = AudioStream::new(StreamHandle(0), Box::new(source), config);

        let mut output = vec![0.0_f32; 44100 * 2]; // 2 seconds of buffer, longer than source.
        stream.read(&mut output, 2.0);
        // Stream should have looped and still be playing.
        assert!(stream.loops_completed > 0);
    }

    #[test]
    fn test_format() {
        let fmt = StreamAudioFormat::cd_quality();
        assert_eq!(fmt.bytes_per_sample(), 4);
        assert_eq!(fmt.bytes_per_second(), 176400);
        assert!((fmt.samples_to_seconds(44100) - 1.0).abs() < 0.001);
    }
}
