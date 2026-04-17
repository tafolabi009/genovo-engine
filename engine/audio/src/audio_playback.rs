// engine/audio/src/audio_playback.rs
//
// Audio playback manager for the Genovo audio module.
//
// Provides sound pool with priority, voice stealing (quietest, oldest,
// farthest), sound categories, fade manager, crossfade, ducking (reduce
// music when dialogue plays), and audio focus.

use std::collections::{HashMap, BinaryHeap, VecDeque};
use std::cmp::Ordering;
use std::fmt;

pub type SoundId = u64;
pub type VoiceId = u64;
pub type CategoryId = u32;
pub type FadeId = u64;

pub const MAX_VOICES: usize = 64;
pub const MAX_CATEGORIES: usize = 16;
pub const MAX_PENDING_SOUNDS: usize = 128;
pub const DEFAULT_MASTER_VOLUME: f32 = 1.0;
pub const FADE_MIN_VOLUME: f32 = 0.001;
pub const DEFAULT_CROSSFADE_DURATION: f32 = 1.0;
pub const DUCKING_FADE_TIME: f32 = 0.3;
pub const DUCKING_VOLUME: f32 = 0.2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoiceStealPolicy { Quietest, Oldest, Farthest, LowestPriority, None }

impl fmt::Display for VoiceStealPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Quietest => write!(f, "Quietest"), Self::Oldest => write!(f, "Oldest"), Self::Farthest => write!(f, "Farthest"), Self::LowestPriority => write!(f, "Lowest Priority"), Self::None => write!(f, "None") }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SoundPriority { Background = 0, Low = 1, Normal = 2, High = 3, Critical = 4, UI = 5, Dialogue = 6 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoiceState { Playing, Paused, FadingIn, FadingOut, Stopped }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FadeType { FadeIn, FadeOut, CrossfadeIn, CrossfadeOut, DuckDown, DuckRestore }

#[derive(Debug, Clone)]
pub struct SoundCategory {
    pub id: CategoryId,
    pub name: String,
    pub volume: f32,
    pub max_voices: usize,
    pub active_voices: usize,
    pub muted: bool,
    pub solo: bool,
    pub steal_policy: VoiceStealPolicy,
    pub duck_targets: Vec<CategoryId>,
    pub duck_amount: f32,
}

impl SoundCategory {
    pub fn new(id: CategoryId, name: &str, max_voices: usize) -> Self {
        Self { id, name: name.to_string(), volume: 1.0, max_voices, active_voices: 0, muted: false, solo: false, steal_policy: VoiceStealPolicy::Quietest, duck_targets: Vec::new(), duck_amount: DUCKING_VOLUME }
    }
    pub fn effective_volume(&self) -> f32 { if self.muted { 0.0 } else { self.volume } }
}

#[derive(Debug, Clone)]
pub struct Voice {
    pub id: VoiceId,
    pub sound_id: SoundId,
    pub category: CategoryId,
    pub priority: SoundPriority,
    pub state: VoiceState,
    pub volume: f32,
    pub effective_volume: f32,
    pub pitch: f32,
    pub pan: f32,
    pub looping: bool,
    pub position: Option<[f32; 3]>,
    pub distance: f32,
    pub start_time: f64,
    pub elapsed: f64,
    pub duration: f64,
    pub fade_volume: f32,
}

impl Voice {
    pub fn new(id: VoiceId, sound_id: SoundId, category: CategoryId, priority: SoundPriority, duration: f64) -> Self {
        Self {
            id, sound_id, category, priority, state: VoiceState::Playing,
            volume: 1.0, effective_volume: 1.0, pitch: 1.0, pan: 0.0,
            looping: false, position: None, distance: 0.0,
            start_time: 0.0, elapsed: 0.0, duration, fade_volume: 1.0,
        }
    }
    pub fn is_playing(&self) -> bool { self.state == VoiceState::Playing || self.state == VoiceState::FadingIn }
    pub fn is_finished(&self) -> bool { !self.looping && self.elapsed >= self.duration && self.state != VoiceState::Paused }
    pub fn compute_effective_volume(&mut self, category_vol: f32, master_vol: f32) {
        self.effective_volume = self.volume * self.fade_volume * category_vol * master_vol;
    }
}

#[derive(Debug, Clone)]
pub struct FadeCommand {
    pub id: FadeId,
    pub voice_id: VoiceId,
    pub fade_type: FadeType,
    pub start_volume: f32,
    pub target_volume: f32,
    pub duration: f32,
    pub elapsed: f32,
    pub completed: bool,
}

impl FadeCommand {
    pub fn new(id: FadeId, voice_id: VoiceId, fade_type: FadeType, from: f32, to: f32, duration: f32) -> Self {
        Self { id, voice_id, fade_type, start_volume: from, target_volume: to, duration: duration.max(0.001), elapsed: 0.0, completed: false }
    }
    pub fn update(&mut self, dt: f32) -> f32 {
        self.elapsed += dt;
        let t = (self.elapsed / self.duration).min(1.0);
        if t >= 1.0 { self.completed = true; }
        // Smooth step interpolation.
        let smooth_t = t * t * (3.0 - 2.0 * t);
        self.start_volume + (self.target_volume - self.start_volume) * smooth_t
    }
}

#[derive(Debug, Clone)]
pub struct PlayRequest {
    pub sound_id: SoundId,
    pub category: CategoryId,
    pub priority: SoundPriority,
    pub volume: f32,
    pub pitch: f32,
    pub pan: f32,
    pub looping: bool,
    pub position: Option<[f32; 3]>,
    pub fade_in: Option<f32>,
    pub delay: f32,
    pub duration: f64,
}

impl Default for PlayRequest {
    fn default() -> Self {
        Self { sound_id: 0, category: 0, priority: SoundPriority::Normal, volume: 1.0, pitch: 1.0, pan: 0.0, looping: false, position: None, fade_in: None, delay: 0.0, duration: 1.0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AudioFocusState { Active, Ducked, Paused, Background }

#[derive(Debug, Clone, Copy, Default)]
pub struct PlaybackStats {
    pub active_voices: u32,
    pub total_voices_played: u64,
    pub voices_stolen: u64,
    pub voices_stopped: u64,
    pub fades_active: u32,
    pub fades_completed: u64,
    pub peak_voices: u32,
    pub ducking_active: bool,
}

pub struct AudioPlaybackManager {
    voices: Vec<Voice>,
    categories: HashMap<CategoryId, SoundCategory>,
    fades: Vec<FadeCommand>,
    master_volume: f32,
    steal_policy: VoiceStealPolicy,
    max_voices: usize,
    stats: PlaybackStats,
    next_voice_id: VoiceId,
    next_fade_id: FadeId,
    time: f64,
    listener_position: [f32; 3],
    focus_state: AudioFocusState,
    ducking_categories: HashMap<CategoryId, f32>,
    pending_requests: VecDeque<(PlayRequest, f32)>,
}

impl AudioPlaybackManager {
    pub fn new(max_voices: usize) -> Self {
        let mut categories = HashMap::new();
        categories.insert(0, SoundCategory::new(0, "Master", max_voices));
        categories.insert(1, SoundCategory::new(1, "Music", 4));
        categories.insert(2, SoundCategory::new(2, "SFX", max_voices / 2));
        categories.insert(3, SoundCategory::new(3, "Dialogue", 2));
        categories.insert(4, SoundCategory::new(4, "Ambient", 8));
        categories.insert(5, SoundCategory::new(5, "UI", 8));

        // Dialogue ducks music.
        if let Some(dialogue) = categories.get_mut(&3) {
            dialogue.duck_targets.push(1);
        }

        Self {
            voices: Vec::with_capacity(max_voices), categories,
            fades: Vec::new(), master_volume: DEFAULT_MASTER_VOLUME,
            steal_policy: VoiceStealPolicy::Quietest,
            max_voices: max_voices.min(MAX_VOICES),
            stats: PlaybackStats::default(), next_voice_id: 1, next_fade_id: 1,
            time: 0.0, listener_position: [0.0; 3], focus_state: AudioFocusState::Active,
            ducking_categories: HashMap::new(), pending_requests: VecDeque::new(),
        }
    }

    pub fn play(&mut self, request: PlayRequest) -> Option<VoiceId> {
        let delay = request.delay;
        if delay > 0.0 {
            self.pending_requests.push_back((request, delay));
            return None;
        }
        self.play_immediate(request)
    }

    fn play_immediate(&mut self, request: PlayRequest) -> Option<VoiceId> {
        // Check category limits.
        if let Some(cat) = self.categories.get(&request.category) {
            if cat.muted { return None; }
            let cat_voices = self.voices.iter().filter(|v| v.category == request.category && v.is_playing()).count();
            if cat_voices >= cat.max_voices {
                // Need to steal.
                if !self.steal_voice(request.category, request.priority) { return None; }
            }
        }

        // Global voice limit.
        if self.voices.len() >= self.max_voices {
            if !self.steal_voice_global(request.priority) { return None; }
        }

        let id = self.next_voice_id; self.next_voice_id += 1;
        let mut voice = Voice::new(id, request.sound_id, request.category, request.priority, request.duration);
        voice.volume = request.volume;
        voice.pitch = request.pitch;
        voice.pan = request.pan;
        voice.looping = request.looping;
        voice.position = request.position;
        voice.start_time = self.time;

        if let Some(fade_dur) = request.fade_in {
            voice.fade_volume = 0.0;
            voice.state = VoiceState::FadingIn;
            self.add_fade(id, FadeType::FadeIn, 0.0, 1.0, fade_dur);
        }

        // Handle ducking.
        if let Some(cat) = self.categories.get(&request.category) {
            let duck_targets = cat.duck_targets.clone();
            let duck_amount = cat.duck_amount;
            for target in duck_targets {
                self.duck_category(target, duck_amount);
            }
        }

        self.voices.push(voice);
        self.stats.total_voices_played += 1;
        self.stats.active_voices = self.voices.len() as u32;
        self.stats.peak_voices = self.stats.peak_voices.max(self.stats.active_voices);
        Some(id)
    }

    fn steal_voice(&mut self, category: CategoryId, min_priority: SoundPriority) -> bool {
        let policy = self.categories.get(&category).map(|c| c.steal_policy).unwrap_or(self.steal_policy);
        self.steal_with_policy(policy, Some(category), min_priority)
    }

    fn steal_voice_global(&mut self, min_priority: SoundPriority) -> bool {
        self.steal_with_policy(self.steal_policy, None, min_priority)
    }

    fn steal_with_policy(&mut self, policy: VoiceStealPolicy, category: Option<CategoryId>, min_priority: SoundPriority) -> bool {
        let candidates: Vec<usize> = self.voices.iter().enumerate()
            .filter(|(_, v)| {
                v.is_playing() && v.priority <= min_priority
                    && category.map_or(true, |c| v.category == c)
            })
            .map(|(i, _)| i)
            .collect();

        if candidates.is_empty() { return false; }

        let steal_idx = match policy {
            VoiceStealPolicy::Quietest => candidates.iter().min_by(|&&a, &&b| self.voices[a].effective_volume.partial_cmp(&self.voices[b].effective_volume).unwrap_or(Ordering::Equal)).copied(),
            VoiceStealPolicy::Oldest => candidates.iter().min_by(|&&a, &&b| self.voices[a].start_time.partial_cmp(&self.voices[b].start_time).unwrap_or(Ordering::Equal)).copied(),
            VoiceStealPolicy::Farthest => candidates.iter().max_by(|&&a, &&b| self.voices[a].distance.partial_cmp(&self.voices[b].distance).unwrap_or(Ordering::Equal)).copied(),
            VoiceStealPolicy::LowestPriority => candidates.iter().min_by_key(|&&i| self.voices[i].priority).copied(),
            VoiceStealPolicy::None => None,
        };

        if let Some(idx) = steal_idx {
            self.voices[idx].state = VoiceState::Stopped;
            self.stats.voices_stolen += 1;
            true
        } else { false }
    }

    pub fn stop(&mut self, voice_id: VoiceId) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == voice_id) {
            v.state = VoiceState::Stopped; self.stats.voices_stopped += 1;
        }
    }

    pub fn stop_with_fade(&mut self, voice_id: VoiceId, duration: f32) {
        let fade_vol = if let Some(v) = self.voices.iter_mut().find(|v| v.id == voice_id) {
            v.state = VoiceState::FadingOut;
            Some(v.fade_volume)
        } else {
            None
        };
        if let Some(fv) = fade_vol {
            self.add_fade(voice_id, FadeType::FadeOut, fv, 0.0, duration);
        }
    }

    pub fn pause(&mut self, voice_id: VoiceId) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == voice_id) { v.state = VoiceState::Paused; }
    }

    pub fn resume(&mut self, voice_id: VoiceId) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == voice_id && v.state == VoiceState::Paused) { v.state = VoiceState::Playing; }
    }

    pub fn crossfade(&mut self, from: VoiceId, to: PlayRequest, duration: f32) -> Option<VoiceId> {
        self.stop_with_fade(from, duration);
        let mut req = to;
        req.fade_in = Some(duration);
        self.play(req)
    }

    fn add_fade(&mut self, voice_id: VoiceId, fade_type: FadeType, from: f32, to: f32, duration: f32) {
        let id = self.next_fade_id; self.next_fade_id += 1;
        self.fades.push(FadeCommand::new(id, voice_id, fade_type, from, to, duration));
    }

    fn duck_category(&mut self, category: CategoryId, amount: f32) {
        self.ducking_categories.insert(category, amount);
        self.stats.ducking_active = true;
        // Apply ducking fade to all voices in this category.
        let voice_ids: Vec<VoiceId> = self.voices.iter().filter(|v| v.category == category && v.is_playing()).map(|v| v.id).collect();
        for vid in voice_ids {
            self.add_fade(vid, FadeType::DuckDown, 1.0, amount, DUCKING_FADE_TIME);
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt as f64;

        // Process delayed requests.
        let mut ready = Vec::new();
        for entry in &mut self.pending_requests {
            entry.1 -= dt;
            if entry.1 <= 0.0 { ready.push(entry.0.clone()); }
        }
        self.pending_requests.retain(|e| e.1 > 0.0);
        for req in ready { self.play_immediate(req); }

        // Update fades.
        for fade in &mut self.fades {
            let vol = fade.update(dt);
            if let Some(voice) = self.voices.iter_mut().find(|v| v.id == fade.voice_id) {
                voice.fade_volume = vol;
                if fade.completed {
                    match fade.fade_type {
                        FadeType::FadeOut | FadeType::CrossfadeOut => { voice.state = VoiceState::Stopped; }
                        FadeType::FadeIn | FadeType::CrossfadeIn => { voice.state = VoiceState::Playing; }
                        FadeType::DuckDown => {}
                        FadeType::DuckRestore => {}
                    }
                    self.stats.fades_completed += 1;
                }
            }
        }
        self.fades.retain(|f| !f.completed);
        self.stats.fades_active = self.fades.len() as u32;

        // Update voices.
        for voice in &mut self.voices {
            if voice.state == VoiceState::Playing || voice.state == VoiceState::FadingIn || voice.state == VoiceState::FadingOut {
                voice.elapsed += dt as f64 * voice.pitch as f64;
                if voice.position.is_some() {
                    let p = voice.position.unwrap();
                    let dx = p[0] - self.listener_position[0];
                    let dy = p[1] - self.listener_position[1];
                    let dz = p[2] - self.listener_position[2];
                    voice.distance = (dx * dx + dy * dy + dz * dz).sqrt();
                }
            }
            let cat_vol = self.categories.get(&voice.category).map(|c| c.effective_volume()).unwrap_or(1.0);
            voice.compute_effective_volume(cat_vol, self.master_volume);
        }

        // Remove finished voices.
        self.voices.retain(|v| v.state != VoiceState::Stopped && !v.is_finished());
        self.stats.active_voices = self.voices.len() as u32;

        // Check if ducking should be restored.
        if self.stats.ducking_active {
            let any_ducking_source = self.categories.values().any(|cat| {
                !cat.duck_targets.is_empty() && self.voices.iter().any(|v| v.category == cat.id && v.is_playing())
            });
            if !any_ducking_source {
                let targets: Vec<CategoryId> = self.ducking_categories.keys().cloned().collect();
                for target in targets {
                    let voice_ids: Vec<VoiceId> = self.voices.iter().filter(|v| v.category == target).map(|v| v.id).collect();
                    for vid in voice_ids { self.add_fade(vid, FadeType::DuckRestore, DUCKING_VOLUME, 1.0, DUCKING_FADE_TIME); }
                }
                self.ducking_categories.clear();
                self.stats.ducking_active = false;
            }
        }
    }

    pub fn set_master_volume(&mut self, vol: f32) { self.master_volume = vol.clamp(0.0, 1.0); }
    pub fn master_volume(&self) -> f32 { self.master_volume }
    pub fn set_category_volume(&mut self, cat: CategoryId, vol: f32) { if let Some(c) = self.categories.get_mut(&cat) { c.volume = vol.clamp(0.0, 1.0); } }
    pub fn mute_category(&mut self, cat: CategoryId, mute: bool) { if let Some(c) = self.categories.get_mut(&cat) { c.muted = mute; } }
    pub fn set_listener_position(&mut self, pos: [f32; 3]) { self.listener_position = pos; }
    pub fn active_voice_count(&self) -> usize { self.voices.len() }
    pub fn stats(&self) -> &PlaybackStats { &self.stats }
    pub fn get_voice(&self, id: VoiceId) -> Option<&Voice> { self.voices.iter().find(|v| v.id == id) }
    pub fn stop_all(&mut self) { for v in &mut self.voices { v.state = VoiceState::Stopped; } }
    pub fn stop_category(&mut self, cat: CategoryId) { for v in &mut self.voices { if v.category == cat { v.state = VoiceState::Stopped; } } }
    pub fn pause_all(&mut self) { for v in &mut self.voices { if v.is_playing() { v.state = VoiceState::Paused; } } }
    pub fn resume_all(&mut self) { for v in &mut self.voices { if v.state == VoiceState::Paused { v.state = VoiceState::Playing; } } }
    pub fn set_focus(&mut self, focus: AudioFocusState) {
        self.focus_state = focus;
        match focus {
            AudioFocusState::Paused => self.pause_all(),
            AudioFocusState::Active => self.resume_all(),
            AudioFocusState::Ducked => { self.master_volume *= 0.3; }
            AudioFocusState::Background => { self.master_volume *= 0.1; }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_play_sound() {
        let mut mgr = AudioPlaybackManager::new(32);
        let req = PlayRequest { sound_id: 1, category: 2, duration: 1.0, ..Default::default() };
        let voice = mgr.play(req);
        assert!(voice.is_some());
        assert_eq!(mgr.active_voice_count(), 1);
    }

    #[test]
    fn test_stop_sound() {
        let mut mgr = AudioPlaybackManager::new(32);
        let req = PlayRequest { sound_id: 1, category: 2, duration: 1.0, ..Default::default() };
        let vid = mgr.play(req).unwrap();
        mgr.stop(vid);
        mgr.update(0.016);
        assert_eq!(mgr.active_voice_count(), 0);
    }

    #[test]
    fn test_voice_stealing() {
        let mut mgr = AudioPlaybackManager::new(4);
        // Fill up voices.
        for i in 0..4 {
            mgr.play(PlayRequest { sound_id: i, category: 0, priority: SoundPriority::Low, duration: 10.0, ..Default::default() });
        }
        assert_eq!(mgr.active_voice_count(), 4);
        // Play a higher priority sound.
        let vid = mgr.play(PlayRequest { sound_id: 100, category: 0, priority: SoundPriority::High, duration: 1.0, ..Default::default() });
        assert!(vid.is_some());
        assert_eq!(mgr.stats().voices_stolen, 1);
    }

    #[test]
    fn test_fade() {
        let mut mgr = AudioPlaybackManager::new(32);
        let vid = mgr.play(PlayRequest { sound_id: 1, category: 2, duration: 5.0, fade_in: Some(1.0), ..Default::default() }).unwrap();
        mgr.update(0.5);
        let voice = mgr.get_voice(vid).unwrap();
        assert!(voice.fade_volume > 0.0 && voice.fade_volume < 1.0);
    }

    #[test]
    fn test_category_mute() {
        let mut mgr = AudioPlaybackManager::new(32);
        mgr.mute_category(2, true);
        let vid = mgr.play(PlayRequest { sound_id: 1, category: 2, duration: 1.0, ..Default::default() });
        assert!(vid.is_none());
    }
}
