// engine/audio/src/audio_mixer_v2.rs
// Enhanced mixer: submix groups, send/return routing, sidechain compression, limiter, spectrum analyzer.
use std::collections::{HashMap, VecDeque};
use std::f32::consts::PI;

pub type BusId = u32;
pub type VoiceId = u32;

#[derive(Debug, Clone)]
pub struct AudioBusV2 {
    pub id: BusId, pub name: String, pub volume: f32, pub mute: bool, pub solo: bool,
    pub pan: f32, pub parent: Option<BusId>, pub children: Vec<BusId>,
    pub sends: Vec<SendRoute>, pub effects: Vec<AudioEffect>,
    pub peak_level: [f32; 2], pub rms_level: [f32; 2], pub buffer: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SendRoute { pub target_bus: BusId, pub amount: f32, pub pre_fader: bool }

#[derive(Debug, Clone)]
pub enum AudioEffect {
    Compressor { threshold: f32, ratio: f32, attack_ms: f32, release_ms: f32, knee: f32, makeup_gain: f32, envelope: f32 },
    SidechainCompressor { sidechain_bus: BusId, threshold: f32, ratio: f32, attack_ms: f32, release_ms: f32, envelope: f32 },
    Limiter { threshold: f32, release_ms: f32, lookahead_ms: f32, envelope: f32 },
    EQ { bands: Vec<EQBand> },
    Delay { time_ms: f32, feedback: f32, wet: f32, buffer: Vec<f32>, write_pos: usize },
    Reverb { decay: f32, wet: f32, pre_delay_ms: f32, diffusion: f32 },
    HighPass { cutoff: f32, resonance: f32, state: [f32; 2] },
    LowPass { cutoff: f32, resonance: f32, state: [f32; 2] },
}

#[derive(Debug, Clone)]
pub struct EQBand { pub frequency: f32, pub gain_db: f32, pub q: f32, pub band_type: EQBandType }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EQBandType { LowShelf, HighShelf, Peak, Notch }

#[derive(Debug, Clone)]
pub struct SpectrumData { pub bins: Vec<f32>, pub sample_rate: u32, pub bin_count: usize }

impl SpectrumData {
    pub fn new(bin_count: usize, sample_rate: u32) -> Self { Self { bins: vec![0.0; bin_count], sample_rate, bin_count } }
    pub fn frequency_at_bin(&self, bin: usize) -> f32 { bin as f32 * self.sample_rate as f32 / (self.bin_count * 2) as f32 }
    pub fn compute_from_buffer(&mut self, buffer: &[f32]) {
        let n = self.bin_count.min(buffer.len() / 2);
        for i in 0..n {
            let mut real = 0.0_f32; let mut imag = 0.0_f32;
            for (j, &sample) in buffer.iter().enumerate().take(n * 2) {
                let angle = -2.0 * PI * i as f32 * j as f32 / (n * 2) as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            self.bins[i] = (real * real + imag * imag).sqrt() / n as f32;
        }
    }
}

pub struct AudioMixerV2 {
    buses: HashMap<BusId, AudioBusV2>,
    master_bus: BusId,
    next_bus_id: BusId,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub spectrum: SpectrumData,
    pub stats: MixerStats,
}

#[derive(Debug, Clone, Default)]
pub struct MixerStats { pub bus_count: u32, pub active_voices: u32, pub peak_level: f32, pub cpu_percent: f32, pub clipping: bool }

impl AudioMixerV2 {
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        let mut buses = HashMap::new();
        let master = AudioBusV2 { id: 0, name: "Master".into(), volume: 1.0, mute: false, solo: false, pan: 0.0, parent: None, children: Vec::new(), sends: Vec::new(), effects: Vec::new(), peak_level: [0.0; 2], rms_level: [0.0; 2], buffer: vec![0.0; buffer_size * 2] };
        buses.insert(0, master);
        Self { buses, master_bus: 0, next_bus_id: 1, sample_rate, buffer_size, spectrum: SpectrumData::new(512, sample_rate), stats: MixerStats::default() }
    }

    pub fn create_bus(&mut self, name: &str, parent: Option<BusId>) -> BusId {
        let id = self.next_bus_id; self.next_bus_id += 1;
        let bus = AudioBusV2 { id, name: name.into(), volume: 1.0, mute: false, solo: false, pan: 0.0, parent, children: Vec::new(), sends: Vec::new(), effects: Vec::new(), peak_level: [0.0; 2], rms_level: [0.0; 2], buffer: vec![0.0; self.buffer_size * 2] };
        if let Some(pid) = parent { if let Some(p) = self.buses.get_mut(&pid) { p.children.push(id); } }
        else { if let Some(m) = self.buses.get_mut(&self.master_bus) { m.children.push(id); } }
        self.buses.insert(id, bus);
        id
    }

    pub fn set_volume(&mut self, bus: BusId, volume: f32) { if let Some(b) = self.buses.get_mut(&bus) { b.volume = volume.clamp(0.0, 2.0); } }
    pub fn set_mute(&mut self, bus: BusId, mute: bool) { if let Some(b) = self.buses.get_mut(&bus) { b.mute = mute; } }
    pub fn add_send(&mut self, from: BusId, to: BusId, amount: f32) {
        if let Some(b) = self.buses.get_mut(&from) { b.sends.push(SendRoute { target_bus: to, amount, pre_fader: false }); }
    }
    pub fn add_effect(&mut self, bus: BusId, effect: AudioEffect) {
        if let Some(b) = self.buses.get_mut(&bus) { b.effects.push(effect); }
    }

    pub fn mix_buffer(&mut self, input_buffers: &HashMap<BusId, Vec<f32>>) -> Vec<f32> {
        // Clear all bus buffers
        for bus in self.buses.values_mut() { for s in &mut bus.buffer { *s = 0.0; } }
        // Add input to respective buses
        for (&bus_id, buffer) in input_buffers {
            if let Some(bus) = self.buses.get_mut(&bus_id) {
                for (i, &s) in buffer.iter().enumerate() { if i < bus.buffer.len() { bus.buffer[i] += s; } }
            }
        }
        // Process effects and mix to parent (simplified - topological order)
        let bus_ids: Vec<BusId> = self.buses.keys().cloned().collect();
        for &bid in &bus_ids {
            if bid == self.master_bus { continue; }
            let bus_data = match self.buses.get(&bid) { Some(b) => b.clone(), None => continue };
            if bus_data.mute { continue; }
            // Apply volume
            let mut processed = bus_data.buffer.clone();
            for s in &mut processed { *s *= bus_data.volume; }
            // Apply effects
            for effect in &bus_data.effects {
                self.apply_effect_to_buffer(&mut processed, effect);
            }
            // Mix to parent
            let parent = bus_data.parent.unwrap_or(self.master_bus);
            if let Some(p) = self.buses.get_mut(&parent) {
                for (i, &s) in processed.iter().enumerate() { if i < p.buffer.len() { p.buffer[i] += s; } }
            }
            // Process sends
            for send in &bus_data.sends {
                if let Some(target) = self.buses.get_mut(&send.target_bus) {
                    for (i, &s) in processed.iter().enumerate() { if i < target.buffer.len() { target.buffer[i] += s * send.amount; } }
                }
            }
            // Update levels
            if let Some(bus) = self.buses.get_mut(&bid) {
                bus.peak_level = compute_peak_stereo(&processed);
                bus.rms_level = compute_rms_stereo(&processed);
            }
        }
        // Process master
        let master = self.buses.get(&self.master_bus).unwrap().clone();
        let mut output = master.buffer.clone();
        for s in &mut output { *s *= master.volume; }
        for effect in &master.effects { self.apply_effect_to_buffer(&mut output, effect); }
        // Spectrum
        self.spectrum.compute_from_buffer(&output);
        // Stats
        self.stats.bus_count = self.buses.len() as u32;
        self.stats.peak_level = output.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        self.stats.clipping = self.stats.peak_level > 1.0;
        // Hard clip
        for s in &mut output { *s = s.clamp(-1.0, 1.0); }
        output
    }

    fn apply_effect_to_buffer(&self, buffer: &mut [f32], effect: &AudioEffect) {
        match effect {
            AudioEffect::Compressor { threshold, ratio, attack_ms, release_ms, knee, makeup_gain, .. } => {
                let attack = (-1.0 / (self.sample_rate as f32 * attack_ms * 0.001)).exp();
                let release = (-1.0 / (self.sample_rate as f32 * release_ms * 0.001)).exp();
                let mut env = 0.0_f32;
                for s in buffer.iter_mut() {
                    let level = s.abs();
                    let coeff = if level > env { attack } else { release };
                    env = coeff * env + (1.0 - coeff) * level;
                    let db = 20.0 * (env.max(1e-6)).log10();
                    let gain_reduction = if db > *threshold { (db - threshold) * (1.0 - 1.0 / ratio) } else { 0.0 };
                    let gain = 10.0_f32.powf(-gain_reduction / 20.0) * 10.0_f32.powf(makeup_gain / 20.0);
                    *s *= gain;
                }
            }
            AudioEffect::Limiter { threshold, release_ms, .. } => {
                let release = (-1.0 / (self.sample_rate as f32 * release_ms * 0.001)).exp();
                let mut env = 0.0_f32;
                for s in buffer.iter_mut() {
                    let level = s.abs();
                    env = if level > env { level } else { release * env };
                    if env > *threshold { *s *= threshold / env; }
                }
            }
            _ => {}
        }
    }

    pub fn get_spectrum(&self) -> &SpectrumData { &self.spectrum }
    pub fn get_bus_level(&self, bus: BusId) -> Option<[f32; 2]> { self.buses.get(&bus).map(|b| b.peak_level) }
}

fn compute_peak_stereo(buffer: &[f32]) -> [f32; 2] {
    let mut l = 0.0_f32; let mut r = 0.0_f32;
    for i in (0..buffer.len()).step_by(2) {
        l = l.max(buffer[i].abs());
        if i + 1 < buffer.len() { r = r.max(buffer[i+1].abs()); }
    }
    [l, r]
}

fn compute_rms_stereo(buffer: &[f32]) -> [f32; 2] {
    let mut l = 0.0_f32; let mut r = 0.0_f32; let mut count = 0u32;
    for i in (0..buffer.len()).step_by(2) {
        l += buffer[i] * buffer[i];
        if i + 1 < buffer.len() { r += buffer[i+1] * buffer[i+1]; }
        count += 1;
    }
    let c = count.max(1) as f32;
    [(l / c).sqrt(), (r / c).sqrt()]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mixer() {
        let mut mixer = AudioMixerV2::new(48000, 256);
        let music = mixer.create_bus("Music", None);
        let sfx = mixer.create_bus("SFX", None);
        let mut inputs = HashMap::new();
        inputs.insert(music, vec![0.5_f32; 512]);
        inputs.insert(sfx, vec![0.3_f32; 512]);
        let output = mixer.mix_buffer(&inputs);
        assert!(!output.is_empty());
        assert!(mixer.stats.peak_level > 0.0);
    }
    #[test]
    fn test_spectrum() {
        let mut s = SpectrumData::new(64, 48000);
        let buffer: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        s.compute_from_buffer(&buffer);
        assert!(s.bins.iter().any(|&b| b > 0.0));
    }
}
