// engine/audio/src/audio_resource.rs
//
// Audio resource management for the Genovo engine.
//
// Manages the lifecycle of audio assets with memory budget awareness:
//
// - Audio clip loading and unloading.
// - Streaming buffer management for large audio files.
// - Audio memory budget tracking and enforcement.
// - Preload lists for level loading.
// - Audio bank system for grouped assets.
// - Reference counting for shared audio data.
// - Async loading with callbacks.
// - Audio format metadata and conversion.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default audio memory budget in bytes (256 MB).
const DEFAULT_MEMORY_BUDGET: usize = 256 * 1024 * 1024;

/// Maximum concurrent streaming buffers.
const MAX_STREAMING_BUFFERS: usize = 32;

/// Default streaming buffer size in samples.
const DEFAULT_STREAM_BUFFER_SIZE: usize = 65536;

/// Minimum audio clip size for streaming (larger clips stream, smaller ones preload).
const STREAM_SIZE_THRESHOLD: usize = 1024 * 1024; // 1 MB

/// Maximum audio banks.
const MAX_BANKS: usize = 64;

/// Cache timeout for unused clips (seconds).
const CACHE_TIMEOUT: f32 = 60.0;

// ---------------------------------------------------------------------------
// Audio Format
// ---------------------------------------------------------------------------

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    /// 8-bit unsigned integer PCM.
    U8,
    /// 16-bit signed integer PCM.
    I16,
    /// 24-bit signed integer PCM.
    I24,
    /// 32-bit signed integer PCM.
    I32,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
}

impl SampleFormat {
    /// Bytes per sample.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I16 => 2,
            Self::I24 => 3,
            Self::I32 => 4,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Bit depth.
    pub fn bit_depth(self) -> u32 {
        match self {
            Self::U8 => 8,
            Self::I16 => 16,
            Self::I24 => 24,
            Self::I32 => 32,
            Self::F32 => 32,
            Self::F64 => 64,
        }
    }
}

/// Audio clip metadata.
#[derive(Debug, Clone)]
pub struct AudioFormatInfo {
    /// Sample rate (e.g., 44100, 48000).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Sample format.
    pub format: SampleFormat,
    /// Total number of samples (per channel).
    pub total_samples: u64,
    /// Duration in seconds.
    pub duration: f64,
    /// Compression codec (None = uncompressed PCM).
    pub codec: Option<String>,
    /// Compressed size in bytes (if compressed).
    pub compressed_size: usize,
    /// Uncompressed size in bytes.
    pub uncompressed_size: usize,
}

impl AudioFormatInfo {
    /// Compute duration from total samples and sample rate.
    pub fn compute_duration(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.total_samples as f64 / self.sample_rate as f64
    }

    /// Bytes per frame (one sample per channel).
    pub fn bytes_per_frame(&self) -> usize {
        self.format.bytes_per_sample() * self.channels as usize
    }

    /// Data rate in bytes per second.
    pub fn data_rate(&self) -> usize {
        self.bytes_per_frame() * self.sample_rate as usize
    }
}

// ---------------------------------------------------------------------------
// Audio Clip Handle
// ---------------------------------------------------------------------------

/// Handle to a loaded audio clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioClipHandle(pub u64);

impl AudioClipHandle {
    /// Invalid handle.
    pub const INVALID: Self = Self(0);

    /// Check validity.
    pub fn is_valid(self) -> bool {
        self.0 != 0
    }
}

// ---------------------------------------------------------------------------
// Audio Clip
// ---------------------------------------------------------------------------

/// Load state of an audio clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipLoadState {
    /// Not loaded.
    Unloaded,
    /// Loading in progress.
    Loading,
    /// Loaded into memory.
    Loaded,
    /// Streaming (only header loaded).
    Streaming,
    /// Load failed.
    Failed,
}

/// An audio clip resource.
#[derive(Debug, Clone)]
pub struct AudioClipResource {
    /// Clip handle.
    pub handle: AudioClipHandle,
    /// Asset path.
    pub path: String,
    /// Format info.
    pub format: AudioFormatInfo,
    /// Load state.
    pub state: ClipLoadState,
    /// PCM data (if fully loaded).
    pub data: Vec<f32>,
    /// Reference count.
    pub ref_count: u32,
    /// Last access time (seconds since engine start).
    pub last_access_time: f64,
    /// Memory used by this clip.
    pub memory_bytes: usize,
    /// Bank this clip belongs to (if any).
    pub bank_id: Option<AudioBankId>,
    /// Whether this clip should use streaming.
    pub use_streaming: bool,
    /// Streaming buffer (if streaming).
    pub stream_buffer: Option<StreamBuffer>,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl AudioClipResource {
    /// Create a new unloaded clip resource.
    pub fn new(handle: AudioClipHandle, path: &str) -> Self {
        Self {
            handle,
            path: path.to_string(),
            format: AudioFormatInfo {
                sample_rate: 44100,
                channels: 2,
                format: SampleFormat::F32,
                total_samples: 0,
                duration: 0.0,
                codec: None,
                compressed_size: 0,
                uncompressed_size: 0,
            },
            state: ClipLoadState::Unloaded,
            data: Vec::new(),
            ref_count: 0,
            last_access_time: 0.0,
            memory_bytes: 0,
            bank_id: None,
            use_streaming: false,
            stream_buffer: None,
            tags: Vec::new(),
        }
    }

    /// Increment reference count.
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count.
    pub fn release(&mut self) {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
    }

    /// Check if clip is loaded and ready for playback.
    pub fn is_ready(&self) -> bool {
        self.state == ClipLoadState::Loaded || self.state == ClipLoadState::Streaming
    }

    /// Whether the clip should be unloaded (unreferenced and timed out).
    pub fn should_unload(&self, current_time: f64) -> bool {
        self.ref_count == 0 && (current_time - self.last_access_time) > CACHE_TIMEOUT as f64
    }

    /// Unload the clip data.
    pub fn unload(&mut self) {
        self.data.clear();
        self.data.shrink_to_fit();
        self.stream_buffer = None;
        self.memory_bytes = 0;
        self.state = ClipLoadState::Unloaded;
    }
}

// ---------------------------------------------------------------------------
// Streaming Buffer
// ---------------------------------------------------------------------------

/// A streaming buffer for playing large audio files.
#[derive(Debug, Clone)]
pub struct StreamBuffer {
    /// Buffer A (double buffering).
    pub buffer_a: Vec<f32>,
    /// Buffer B.
    pub buffer_b: Vec<f32>,
    /// Which buffer is currently active (0 or 1).
    pub active_buffer: u32,
    /// Read position within the active buffer.
    pub read_position: usize,
    /// File position (sample offset in the full clip).
    pub file_position: u64,
    /// Whether buffer B needs to be filled.
    pub needs_fill: bool,
    /// Buffer size in samples.
    pub buffer_size: usize,
    /// Whether the stream has reached the end.
    pub end_of_stream: bool,
    /// Whether to loop.
    pub looping: bool,
}

impl StreamBuffer {
    /// Create a new streaming buffer.
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer_a: vec![0.0; buffer_size],
            buffer_b: vec![0.0; buffer_size],
            active_buffer: 0,
            read_position: 0,
            file_position: 0,
            needs_fill: true,
            buffer_size,
            end_of_stream: false,
            looping: false,
        }
    }

    /// Get the active buffer.
    pub fn active(&self) -> &[f32] {
        if self.active_buffer == 0 { &self.buffer_a } else { &self.buffer_b }
    }

    /// Get the inactive buffer for filling.
    pub fn inactive_mut(&mut self) -> &mut [f32] {
        if self.active_buffer == 0 { &mut self.buffer_b } else { &mut self.buffer_a }
    }

    /// Swap buffers.
    pub fn swap(&mut self) {
        self.active_buffer = 1 - self.active_buffer;
        self.read_position = 0;
        self.needs_fill = true;
    }

    /// Read samples from the stream. Returns the number of samples read.
    pub fn read(&mut self, output: &mut [f32]) -> usize {
        let active = if self.active_buffer == 0 { &self.buffer_a } else { &self.buffer_b };
        let available = active.len() - self.read_position;
        let to_read = output.len().min(available);

        output[..to_read].copy_from_slice(&active[self.read_position..self.read_position + to_read]);
        self.read_position += to_read;

        if self.read_position >= active.len() {
            self.swap();
        }

        to_read
    }

    /// Total memory used by the stream buffers.
    pub fn memory_bytes(&self) -> usize {
        (self.buffer_a.len() + self.buffer_b.len()) * std::mem::size_of::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Audio Bank
// ---------------------------------------------------------------------------

/// Unique identifier for an audio bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioBankId(pub u32);

/// An audio bank groups related clips for batch loading/unloading.
#[derive(Debug, Clone)]
pub struct AudioBank {
    /// Bank identifier.
    pub id: AudioBankId,
    /// Bank name.
    pub name: String,
    /// Clips in this bank.
    pub clips: Vec<AudioClipHandle>,
    /// Whether this bank is currently loaded.
    pub loaded: bool,
    /// Total memory used by this bank.
    pub memory_bytes: usize,
    /// Priority (higher = loaded first).
    pub priority: i32,
    /// Whether to keep this bank always loaded.
    pub persistent: bool,
}

impl AudioBank {
    /// Create a new bank.
    pub fn new(id: AudioBankId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            clips: Vec::new(),
            loaded: false,
            memory_bytes: 0,
            priority: 0,
            persistent: false,
        }
    }

    /// Add a clip to this bank.
    pub fn add_clip(&mut self, handle: AudioClipHandle) {
        if !self.clips.contains(&handle) {
            self.clips.push(handle);
        }
    }

    /// Number of clips.
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }
}

// ---------------------------------------------------------------------------
// Preload List
// ---------------------------------------------------------------------------

/// A preload list for batch loading audio clips.
#[derive(Debug, Clone)]
pub struct PreloadList {
    /// List name (e.g., level name).
    pub name: String,
    /// Clip paths to preload.
    pub clip_paths: Vec<String>,
    /// Bank IDs to preload.
    pub bank_ids: Vec<AudioBankId>,
    /// Priority.
    pub priority: i32,
    /// Whether loading is complete.
    pub loaded: bool,
    /// Loading progress (0.0 to 1.0).
    pub progress: f32,
}

impl PreloadList {
    /// Create a new preload list.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            clip_paths: Vec::new(),
            bank_ids: Vec::new(),
            priority: 0,
            loaded: false,
            progress: 0.0,
        }
    }

    /// Add a clip path.
    pub fn add_clip(&mut self, path: &str) {
        self.clip_paths.push(path.to_string());
    }

    /// Add a bank.
    pub fn add_bank(&mut self, bank_id: AudioBankId) {
        self.bank_ids.push(bank_id);
    }
}

// ---------------------------------------------------------------------------
// Audio Resource Manager
// ---------------------------------------------------------------------------

/// Manages all audio resources.
#[derive(Debug)]
pub struct AudioResourceManager {
    /// Loaded audio clips.
    pub clips: HashMap<AudioClipHandle, AudioClipResource>,
    /// Path-to-handle lookup.
    pub path_lookup: HashMap<String, AudioClipHandle>,
    /// Audio banks.
    pub banks: HashMap<AudioBankId, AudioBank>,
    /// Preload lists.
    pub preload_lists: Vec<PreloadList>,
    /// Next clip handle.
    next_handle: u64,
    /// Next bank ID.
    next_bank_id: u32,
    /// Memory budget in bytes.
    pub memory_budget: usize,
    /// Current memory usage in bytes.
    pub memory_used: usize,
    /// Current time.
    pub current_time: f64,
    /// Events.
    pub events: Vec<AudioResourceEvent>,
    /// Statistics.
    pub stats: AudioResourceStats,
    /// Streaming buffer size.
    pub stream_buffer_size: usize,
}

/// Events from the audio resource manager.
#[derive(Debug, Clone)]
pub enum AudioResourceEvent {
    /// Clip loaded.
    ClipLoaded { handle: AudioClipHandle, path: String },
    /// Clip unloaded.
    ClipUnloaded { handle: AudioClipHandle },
    /// Clip load failed.
    ClipLoadFailed { path: String, error: String },
    /// Bank loaded.
    BankLoaded { bank_id: AudioBankId, name: String },
    /// Bank unloaded.
    BankUnloaded { bank_id: AudioBankId },
    /// Memory budget exceeded.
    MemoryBudgetExceeded { used: usize, budget: usize },
    /// Preload list completed.
    PreloadComplete { name: String },
}

impl AudioResourceManager {
    /// Create a new audio resource manager.
    pub fn new() -> Self {
        Self {
            clips: HashMap::new(),
            path_lookup: HashMap::new(),
            banks: HashMap::new(),
            preload_lists: Vec::new(),
            next_handle: 1,
            next_bank_id: 1,
            memory_budget: DEFAULT_MEMORY_BUDGET,
            memory_used: 0,
            current_time: 0.0,
            events: Vec::new(),
            stats: AudioResourceStats::default(),
            stream_buffer_size: DEFAULT_STREAM_BUFFER_SIZE,
        }
    }

    /// Load an audio clip from a path. Returns the handle.
    pub fn load_clip(&mut self, path: &str) -> AudioClipHandle {
        // Check if already loaded.
        if let Some(&handle) = self.path_lookup.get(path) {
            if let Some(clip) = self.clips.get_mut(&handle) {
                clip.add_ref();
                clip.last_access_time = self.current_time;
            }
            return handle;
        }

        let handle = AudioClipHandle(self.next_handle);
        self.next_handle += 1;

        let mut clip = AudioClipResource::new(handle, path);
        clip.ref_count = 1;
        clip.last_access_time = self.current_time;
        clip.state = ClipLoadState::Loaded; // Simplified: mark as loaded.

        self.path_lookup.insert(path.to_string(), handle);
        self.clips.insert(handle, clip);

        self.events.push(AudioResourceEvent::ClipLoaded {
            handle,
            path: path.to_string(),
        });

        self.stats.clips_loaded += 1;
        handle
    }

    /// Unload an audio clip.
    pub fn unload_clip(&mut self, handle: AudioClipHandle) {
        if let Some(clip) = self.clips.get_mut(&handle) {
            let memory_freed = clip.memory_bytes;
            clip.unload();
            self.memory_used = self.memory_used.saturating_sub(memory_freed);
            self.path_lookup.remove(&clip.path);
            self.events.push(AudioResourceEvent::ClipUnloaded { handle });
            self.stats.clips_unloaded += 1;
        }
        self.clips.remove(&handle);
    }

    /// Get a clip by handle.
    pub fn get_clip(&self, handle: AudioClipHandle) -> Option<&AudioClipResource> {
        self.clips.get(&handle)
    }

    /// Get a mutable clip.
    pub fn get_clip_mut(&mut self, handle: AudioClipHandle) -> Option<&mut AudioClipResource> {
        self.clips.get_mut(&handle)
    }

    /// Create an audio bank.
    pub fn create_bank(&mut self, name: &str) -> AudioBankId {
        let id = AudioBankId(self.next_bank_id);
        self.next_bank_id += 1;
        self.banks.insert(id, AudioBank::new(id, name));
        id
    }

    /// Load all clips in a bank.
    pub fn load_bank(&mut self, bank_id: AudioBankId) {
        if let Some(bank) = self.banks.get(&bank_id).cloned() {
            for &clip_handle in &bank.clips {
                if let Some(clip) = self.clips.get_mut(&clip_handle) {
                    if clip.state == ClipLoadState::Unloaded {
                        clip.state = ClipLoadState::Loaded;
                        clip.add_ref();
                    }
                }
            }
            if let Some(bank) = self.banks.get_mut(&bank_id) {
                bank.loaded = true;
            }
            self.events.push(AudioResourceEvent::BankLoaded {
                bank_id,
                name: bank.name.clone(),
            });
        }
    }

    /// Unload all clips in a bank.
    pub fn unload_bank(&mut self, bank_id: AudioBankId) {
        if let Some(bank) = self.banks.get(&bank_id).cloned() {
            if bank.persistent { return; }
            for &clip_handle in &bank.clips {
                if let Some(clip) = self.clips.get_mut(&clip_handle) {
                    clip.release();
                }
            }
            if let Some(bank) = self.banks.get_mut(&bank_id) {
                bank.loaded = false;
            }
            self.events.push(AudioResourceEvent::BankUnloaded { bank_id });
        }
    }

    /// Update the resource manager (cleanup, memory management).
    pub fn update(&mut self, dt: f32) {
        self.current_time += dt as f64;

        // Unload expired clips.
        let to_unload: Vec<AudioClipHandle> = self.clips.iter()
            .filter(|(_, c)| c.should_unload(self.current_time))
            .map(|(&h, _)| h)
            .collect();

        for handle in to_unload {
            self.unload_clip(handle);
        }

        // Update memory tracking.
        self.memory_used = self.clips.values().map(|c| c.memory_bytes).sum();

        // Check budget.
        if self.memory_used > self.memory_budget {
            self.events.push(AudioResourceEvent::MemoryBudgetExceeded {
                used: self.memory_used,
                budget: self.memory_budget,
            });
            self.evict_least_recent();
        }

        // Update stats.
        self.stats.total_clips = self.clips.len() as u32;
        self.stats.loaded_clips = self.clips.values().filter(|c| c.is_ready()).count() as u32;
        self.stats.memory_used = self.memory_used;
        self.stats.memory_budget = self.memory_budget;
        self.stats.bank_count = self.banks.len() as u32;
    }

    /// Evict the least recently used clips to free memory.
    fn evict_least_recent(&mut self) {
        let mut candidates: Vec<(AudioClipHandle, f64)> = self.clips.iter()
            .filter(|(_, c)| c.ref_count == 0 && c.is_ready())
            .map(|(&h, c)| (h, c.last_access_time))
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (handle, _) in candidates {
            if self.memory_used <= self.memory_budget {
                break;
            }
            self.unload_clip(handle);
        }
    }

    /// Set the memory budget.
    pub fn set_memory_budget(&mut self, budget_bytes: usize) {
        self.memory_budget = budget_bytes;
    }

    /// Get the memory usage ratio (0.0 to 1.0+).
    pub fn memory_usage_ratio(&self) -> f32 {
        if self.memory_budget == 0 { return 0.0; }
        self.memory_used as f32 / self.memory_budget as f32
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<AudioResourceEvent> {
        std::mem::take(&mut self.events)
    }
}

/// Audio resource statistics.
#[derive(Debug, Clone, Default)]
pub struct AudioResourceStats {
    /// Total registered clips.
    pub total_clips: u32,
    /// Currently loaded clips.
    pub loaded_clips: u32,
    /// Clips loaded since start.
    pub clips_loaded: u64,
    /// Clips unloaded since start.
    pub clips_unloaded: u64,
    /// Current memory usage.
    pub memory_used: usize,
    /// Memory budget.
    pub memory_budget: usize,
    /// Number of banks.
    pub bank_count: u32,
    /// Number of streaming buffers active.
    pub streaming_buffers: u32,
}
