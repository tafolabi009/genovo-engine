// engine/audio/src/audio_engine.rs
//
// Master audio engine: initialize audio device, manage all audio subsystems
// (mixer, spatial, DSP, music, ambience), audio thread, audio update pipeline,
// master volume, audio pause/resume, audio statistics.
//
// The AudioEngine is the top-level orchestrator for all audio in the engine.
// It manages the audio device lifecycle, routes audio through the mixing and
// spatial pipelines, and provides a unified API for playing sounds, music,
// and ambient audio.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Audio format
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat { I16, I24, I32, F32 }

impl SampleFormat {
    pub fn bytes_per_sample(&self) -> u32 {
        match self { Self::I16 => 2, Self::I24 => 3, Self::I32 => 4, Self::F32 => 4 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioDeviceConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub buffer_size: u32,
    pub format: SampleFormat,
    pub latency_ms: f32,
}

impl Default for AudioDeviceConfig {
    fn default() -> Self {
        Self { sample_rate: 44100, channels: 2, buffer_size: 1024, format: SampleFormat::F32, latency_ms: 23.0 }
    }
}

impl AudioDeviceConfig {
    pub fn high_quality() -> Self { Self { sample_rate: 48000, buffer_size: 512, latency_ms: 10.7, ..Default::default() } }
    pub fn low_latency() -> Self { Self { sample_rate: 48000, buffer_size: 256, latency_ms: 5.3, ..Default::default() } }
}

// ---------------------------------------------------------------------------
// Audio handle / clip
// ---------------------------------------------------------------------------

pub type SoundHandle = u64;
pub type SoundId = u64;

#[derive(Debug, Clone)]
pub struct AudioClipData {
    pub id: SoundId,
    pub name: String,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_secs: f32,
    pub looping: bool,
}

impl AudioClipData {
    pub fn new(id: SoundId, name: &str, samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        let duration = if sample_rate > 0 && channels > 0 {
            samples.len() as f32 / (sample_rate as f32 * channels as f32)
        } else { 0.0 };
        Self { id, name: name.to_string(), samples, sample_rate, channels, duration_secs: duration, looping: false }
    }
    pub fn sample_count(&self) -> usize { self.samples.len() }
    pub fn frame_count(&self) -> usize { if self.channels > 0 { self.samples.len() / self.channels as usize } else { 0 } }
}

// ---------------------------------------------------------------------------
// Voice (playing instance)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaybackState { Playing, Paused, Stopped, FadingIn, FadingOut }

#[derive(Debug, Clone)]
pub struct AudioVoice {
    pub handle: SoundHandle,
    pub clip_id: SoundId,
    pub state: PlaybackState,
    pub volume: f32,
    pub pitch: f32,
    pub pan: f32,
    pub position_samples: usize,
    pub looping: bool,
    pub bus: AudioBusId,
    pub priority: u8,
    pub fade_volume: f32,
    pub fade_target: f32,
    pub fade_speed: f32,
    pub spatial: bool,
    pub world_position: [f32; 3],
    pub max_distance: f32,
    pub min_distance: f32,
    pub creation_time: f64,
}

impl AudioVoice {
    pub fn new(handle: SoundHandle, clip_id: SoundId) -> Self {
        Self {
            handle, clip_id, state: PlaybackState::Playing,
            volume: 1.0, pitch: 1.0, pan: 0.0,
            position_samples: 0, looping: false,
            bus: AudioBusId::Master, priority: 128,
            fade_volume: 1.0, fade_target: 1.0, fade_speed: 0.0,
            spatial: false, world_position: [0.0; 3],
            max_distance: 50.0, min_distance: 1.0, creation_time: 0.0,
        }
    }

    pub fn effective_volume(&self) -> f32 { self.volume * self.fade_volume }
    pub fn is_playing(&self) -> bool { self.state == PlaybackState::Playing || self.state == PlaybackState::FadingIn }
    pub fn is_stopped(&self) -> bool { self.state == PlaybackState::Stopped }
}

// ---------------------------------------------------------------------------
// Audio bus
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioBusId(pub u32);

#[derive(Debug, Clone)]
pub struct AudioBus {
    pub id: AudioBusId,
    pub name: String,
    pub volume: f32,
    pub muted: bool,
    pub parent: Option<AudioBusId>,
    pub effects: Vec<BusEffect>,
    pub voice_count: u32,
    pub peak_level: [f32; 2],
}

impl AudioBus {
    pub fn new(id: AudioBusId, name: &str) -> Self {
        Self { id, name: name.to_string(), volume: 1.0, muted: false, parent: None, effects: Vec::new(), voice_count: 0, peak_level: [0.0; 2] }
    }
    pub fn effective_volume(&self) -> f32 { if self.muted { 0.0 } else { self.volume } }
}

impl AudioBusId {
    pub const Master: AudioBusId = AudioBusId(0);
    pub const Music: AudioBusId = AudioBusId(1);
    pub const SFX: AudioBusId = AudioBusId(2);
    pub const Voice: AudioBusId = AudioBusId(3);
    pub const Ambient: AudioBusId = AudioBusId(4);
    pub const UI: AudioBusId = AudioBusId(5);
}

#[derive(Debug, Clone)]
pub struct BusEffect {
    pub name: String,
    pub enabled: bool,
    pub params: HashMap<String, f32>,
}

// ---------------------------------------------------------------------------
// Listener
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AudioListener {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub velocity: [f32; 3],
}

impl Default for AudioListener {
    fn default() -> Self {
        Self { position: [0.0; 3], forward: [0.0, 0.0, -1.0], up: [0.0, 1.0, 0.0], velocity: [0.0; 3] }
    }
}

// ---------------------------------------------------------------------------
// Play parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PlayParams {
    pub volume: f32,
    pub pitch: f32,
    pub pan: f32,
    pub looping: bool,
    pub bus: AudioBusId,
    pub priority: u8,
    pub fade_in: f32,
    pub spatial: bool,
    pub position: [f32; 3],
    pub min_distance: f32,
    pub max_distance: f32,
    pub delay: f32,
}

impl Default for PlayParams {
    fn default() -> Self {
        Self {
            volume: 1.0, pitch: 1.0, pan: 0.0, looping: false,
            bus: AudioBusId::SFX, priority: 128, fade_in: 0.0,
            spatial: false, position: [0.0; 3], min_distance: 1.0, max_distance: 50.0, delay: 0.0,
        }
    }
}

impl PlayParams {
    pub fn spatial(mut self, pos: [f32; 3]) -> Self { self.spatial = true; self.position = pos; self }
    pub fn with_volume(mut self, v: f32) -> Self { self.volume = v; self }
    pub fn with_pitch(mut self, p: f32) -> Self { self.pitch = p; self }
    pub fn with_bus(mut self, bus: AudioBusId) -> Self { self.bus = bus; self }
    pub fn looped(mut self) -> Self { self.looping = true; self }
}

// ---------------------------------------------------------------------------
// Audio statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct AudioStats {
    pub active_voices: u32,
    pub max_voices: u32,
    pub total_clips_loaded: u32,
    pub total_memory_bytes: u64,
    pub cpu_usage_percent: f32,
    pub buffer_underruns: u32,
    pub sample_rate: u32,
    pub latency_ms: f32,
    pub peak_level: [f32; 2],
    pub voices_per_bus: HashMap<String, u32>,
    pub uptime_secs: f64,
}

// ---------------------------------------------------------------------------
// Audio engine
// ---------------------------------------------------------------------------

pub struct AudioEngine {
    config: AudioDeviceConfig,
    clips: HashMap<SoundId, Arc<AudioClipData>>,
    voices: Vec<AudioVoice>,
    buses: HashMap<AudioBusId, AudioBus>,
    listener: AudioListener,
    master_volume: f32,
    max_voices: u32,
    paused: bool,
    next_handle: SoundHandle,
    next_clip_id: SoundId,
    start_time: Instant,
    elapsed: f64,
    pub stats: AudioStats,
    voice_steal_policy: VoiceStealPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceStealPolicy { None, Oldest, Quietest, LowestPriority, Farthest }

impl AudioEngine {
    pub fn new(config: AudioDeviceConfig) -> Self {
        let mut buses = HashMap::new();
        buses.insert(AudioBusId::Master, AudioBus::new(AudioBusId::Master, "Master"));
        buses.insert(AudioBusId::Music, AudioBus { parent: Some(AudioBusId::Master), ..AudioBus::new(AudioBusId::Music, "Music") });
        buses.insert(AudioBusId::SFX, AudioBus { parent: Some(AudioBusId::Master), ..AudioBus::new(AudioBusId::SFX, "SFX") });
        buses.insert(AudioBusId::Voice, AudioBus { parent: Some(AudioBusId::Master), ..AudioBus::new(AudioBusId::Voice, "Voice") });
        buses.insert(AudioBusId::Ambient, AudioBus { parent: Some(AudioBusId::Master), ..AudioBus::new(AudioBusId::Ambient, "Ambient") });
        buses.insert(AudioBusId::UI, AudioBus { parent: Some(AudioBusId::Master), ..AudioBus::new(AudioBusId::UI, "UI") });

        Self {
            config, clips: HashMap::new(), voices: Vec::new(), buses,
            listener: AudioListener::default(), master_volume: 1.0,
            max_voices: 64, paused: false, next_handle: 1, next_clip_id: 1,
            start_time: Instant::now(), elapsed: 0.0,
            stats: AudioStats::default(), voice_steal_policy: VoiceStealPolicy::Quietest,
        }
    }

    pub fn initialize(&mut self) -> Result<(), String> {
        self.stats.sample_rate = self.config.sample_rate;
        self.stats.latency_ms = self.config.latency_ms;
        self.stats.max_voices = self.max_voices;
        Ok(())
    }

    pub fn shutdown(&mut self) {
        self.stop_all();
        self.clips.clear();
    }

    // --- Clip management ---
    pub fn load_clip(&mut self, name: &str, samples: Vec<f32>, sample_rate: u32, channels: u16) -> SoundId {
        let id = self.next_clip_id; self.next_clip_id += 1;
        let clip = AudioClipData::new(id, name, samples, sample_rate, channels);
        self.clips.insert(id, Arc::new(clip));
        id
    }

    pub fn unload_clip(&mut self, id: SoundId) {
        self.voices.retain(|v| v.clip_id != id);
        self.clips.remove(&id);
    }

    pub fn get_clip(&self, id: SoundId) -> Option<&Arc<AudioClipData>> { self.clips.get(&id) }

    // --- Playback ---
    pub fn play(&mut self, clip_id: SoundId, params: PlayParams) -> Option<SoundHandle> {
        if self.paused { return None; }
        if !self.clips.contains_key(&clip_id) { return None; }

        // Voice stealing if at capacity.
        if self.voices.len() >= self.max_voices as usize {
            self.steal_voice();
        }
        if self.voices.len() >= self.max_voices as usize { return None; }

        let handle = self.next_handle; self.next_handle += 1;
        let mut voice = AudioVoice::new(handle, clip_id);
        voice.volume = params.volume;
        voice.pitch = params.pitch;
        voice.pan = params.pan;
        voice.looping = params.looping;
        voice.bus = params.bus;
        voice.priority = params.priority;
        voice.spatial = params.spatial;
        voice.world_position = params.position;
        voice.min_distance = params.min_distance;
        voice.max_distance = params.max_distance;
        voice.creation_time = self.elapsed;

        if params.fade_in > 0.0 {
            voice.state = PlaybackState::FadingIn;
            voice.fade_volume = 0.0;
            voice.fade_target = 1.0;
            voice.fade_speed = 1.0 / params.fade_in;
        }

        self.voices.push(voice);
        Some(handle)
    }

    pub fn play_simple(&mut self, clip_id: SoundId) -> Option<SoundHandle> { self.play(clip_id, PlayParams::default()) }

    pub fn play_3d(&mut self, clip_id: SoundId, position: [f32; 3], volume: f32) -> Option<SoundHandle> {
        self.play(clip_id, PlayParams::default().spatial(position).with_volume(volume))
    }

    pub fn stop(&mut self, handle: SoundHandle) {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.handle == handle) { voice.state = PlaybackState::Stopped; }
    }

    pub fn stop_with_fade(&mut self, handle: SoundHandle, fade_out_secs: f32) {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.handle == handle) {
            voice.state = PlaybackState::FadingOut;
            voice.fade_target = 0.0;
            voice.fade_speed = if fade_out_secs > 0.0 { 1.0 / fade_out_secs } else { 100.0 };
        }
    }

    pub fn pause(&mut self, handle: SoundHandle) {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.handle == handle) { voice.state = PlaybackState::Paused; }
    }

    pub fn resume(&mut self, handle: SoundHandle) {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.handle == handle && v.state == PlaybackState::Paused) { voice.state = PlaybackState::Playing; }
    }

    pub fn stop_all(&mut self) { for voice in &mut self.voices { voice.state = PlaybackState::Stopped; } }

    pub fn set_volume(&mut self, handle: SoundHandle, volume: f32) {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.handle == handle) { voice.volume = volume.clamp(0.0, 2.0); }
    }

    pub fn set_pitch(&mut self, handle: SoundHandle, pitch: f32) {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.handle == handle) { voice.pitch = pitch.clamp(0.1, 4.0); }
    }

    pub fn set_position_3d(&mut self, handle: SoundHandle, position: [f32; 3]) {
        if let Some(voice) = self.voices.iter_mut().find(|v| v.handle == handle) { voice.world_position = position; }
    }

    pub fn is_playing(&self, handle: SoundHandle) -> bool {
        self.voices.iter().any(|v| v.handle == handle && v.is_playing())
    }

    // --- Global controls ---
    pub fn set_master_volume(&mut self, volume: f32) { self.master_volume = volume.clamp(0.0, 1.0); }
    pub fn master_volume(&self) -> f32 { self.master_volume }

    pub fn pause_all(&mut self) { self.paused = true; for v in &mut self.voices { if v.is_playing() { v.state = PlaybackState::Paused; } } }
    pub fn resume_all(&mut self) { self.paused = false; for v in &mut self.voices { if v.state == PlaybackState::Paused { v.state = PlaybackState::Playing; } } }
    pub fn is_paused(&self) -> bool { self.paused }

    // --- Bus controls ---
    pub fn set_bus_volume(&mut self, bus: AudioBusId, volume: f32) {
        if let Some(b) = self.buses.get_mut(&bus) { b.volume = volume.clamp(0.0, 2.0); }
    }

    pub fn set_bus_muted(&mut self, bus: AudioBusId, muted: bool) {
        if let Some(b) = self.buses.get_mut(&bus) { b.muted = muted; }
    }

    pub fn get_bus(&self, bus: AudioBusId) -> Option<&AudioBus> { self.buses.get(&bus) }

    // --- Listener ---
    pub fn set_listener(&mut self, listener: AudioListener) { self.listener = listener; }
    pub fn listener(&self) -> &AudioListener { &self.listener }

    // --- Update ---
    pub fn update(&mut self, dt: f32) {
        self.elapsed += dt as f64;

        // Update fades.
        for voice in &mut self.voices {
            match voice.state {
                PlaybackState::FadingIn => {
                    voice.fade_volume += voice.fade_speed * dt;
                    if voice.fade_volume >= voice.fade_target { voice.fade_volume = voice.fade_target; voice.state = PlaybackState::Playing; }
                }
                PlaybackState::FadingOut => {
                    voice.fade_volume -= voice.fade_speed * dt;
                    if voice.fade_volume <= 0.0 { voice.fade_volume = 0.0; voice.state = PlaybackState::Stopped; }
                }
                _ => {}
            }

            // Advance playback position (simplified).
            if voice.is_playing() {
                if let Some(clip) = self.clips.get(&voice.clip_id) {
                    let advance = (self.config.sample_rate as f32 * voice.pitch * dt) as usize * clip.channels as usize;
                    voice.position_samples += advance;
                    if voice.position_samples >= clip.samples.len() {
                        if voice.looping { voice.position_samples %= clip.samples.len(); }
                        else { voice.state = PlaybackState::Stopped; }
                    }
                }
            }
        }

        // Remove stopped voices.
        self.voices.retain(|v| !v.is_stopped());

        // Update bus stats.
        for bus in self.buses.values_mut() { bus.voice_count = 0; }
        for voice in &self.voices {
            if let Some(bus) = self.buses.get_mut(&voice.bus) { bus.voice_count += 1; }
        }

        // Update stats.
        self.stats.active_voices = self.voices.len() as u32;
        self.stats.total_clips_loaded = self.clips.len() as u32;
        self.stats.total_memory_bytes = self.clips.values().map(|c| (c.samples.len() * 4) as u64).sum();
        self.stats.uptime_secs = self.start_time.elapsed().as_secs_f64();
        self.stats.voices_per_bus.clear();
        for bus in self.buses.values() { self.stats.voices_per_bus.insert(bus.name.clone(), bus.voice_count); }
    }

    fn steal_voice(&mut self) {
        let steal_idx = match self.voice_steal_policy {
            VoiceStealPolicy::None => return,
            VoiceStealPolicy::Oldest => self.voices.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.creation_time.partial_cmp(&b.creation_time).unwrap()).map(|(i, _)| i),
            VoiceStealPolicy::Quietest => self.voices.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.effective_volume().partial_cmp(&b.effective_volume()).unwrap()).map(|(i, _)| i),
            VoiceStealPolicy::LowestPriority => self.voices.iter().enumerate()
                .min_by_key(|(_, v)| v.priority).map(|(i, _)| i),
            VoiceStealPolicy::Farthest => {
                let lp = self.listener.position;
                self.voices.iter().enumerate()
                    .filter(|(_, v)| v.spatial)
                    .max_by(|(_, a), (_, b)| {
                        let da = dist_sq(a.world_position, lp);
                        let db = dist_sq(b.world_position, lp);
                        da.partial_cmp(&db).unwrap()
                    }).map(|(i, _)| i)
            }
        };

        if let Some(idx) = steal_idx { self.voices[idx].state = PlaybackState::Stopped; self.voices.remove(idx); }
    }

    pub fn active_voice_count(&self) -> u32 { self.voices.len() as u32 }
    pub fn max_voice_count(&self) -> u32 { self.max_voices }
    pub fn set_max_voices(&mut self, max: u32) { self.max_voices = max; self.stats.max_voices = max; }
}

fn dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0]; let dy = a[1] - b[1]; let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> AudioEngine {
        let mut engine = AudioEngine::new(AudioDeviceConfig::default());
        engine.initialize().unwrap();
        engine
    }

    fn make_clip(engine: &mut AudioEngine) -> SoundId {
        let samples: Vec<f32> = (0..44100).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin()).collect();
        engine.load_clip("test_tone", samples, 44100, 1)
    }

    #[test]
    fn test_load_and_play() {
        let mut engine = make_engine();
        let clip = make_clip(&mut engine);
        let handle = engine.play_simple(clip);
        assert!(handle.is_some());
        assert!(engine.is_playing(handle.unwrap()));
    }

    #[test]
    fn test_stop() {
        let mut engine = make_engine();
        let clip = make_clip(&mut engine);
        let handle = engine.play_simple(clip).unwrap();
        engine.stop(handle);
        engine.update(0.016);
        assert!(!engine.is_playing(handle));
    }

    #[test]
    fn test_master_volume() {
        let mut engine = make_engine();
        engine.set_master_volume(0.5);
        assert!((engine.master_volume() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_pause_resume() {
        let mut engine = make_engine();
        let clip = make_clip(&mut engine);
        let handle = engine.play_simple(clip).unwrap();
        engine.pause(handle);
        assert!(!engine.is_playing(handle));
        engine.resume(handle);
        assert!(engine.is_playing(handle));
    }

    #[test]
    fn test_voice_stealing() {
        let mut engine = make_engine();
        engine.set_max_voices(2);
        let clip = make_clip(&mut engine);
        engine.play(clip, PlayParams::default().with_volume(1.0)).unwrap();
        engine.play(clip, PlayParams::default().with_volume(0.5)).unwrap();
        let h3 = engine.play(clip, PlayParams::default().with_volume(0.8));
        assert!(h3.is_some()); // Should steal the quietest voice
        assert_eq!(engine.active_voice_count(), 2);
    }

    #[test]
    fn test_bus_volume() {
        let mut engine = make_engine();
        engine.set_bus_volume(AudioBusId::Music, 0.3);
        let bus = engine.get_bus(AudioBusId::Music).unwrap();
        assert!((bus.volume - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_stats() {
        let mut engine = make_engine();
        let clip = make_clip(&mut engine);
        engine.play_simple(clip);
        engine.update(0.016);
        assert_eq!(engine.stats.active_voices, 1);
        assert_eq!(engine.stats.total_clips_loaded, 1);
    }
}
