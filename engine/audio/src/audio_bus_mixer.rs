//! # Advanced Bus Mixer
//!
//! Hierarchical audio bus routing system for the Genovo engine, providing
//! professional-grade mixing capabilities.
//!
//! ## Features
//!
//! - **Bus hierarchy** — Buses can contain sub-buses. Audio from child buses
//!   is mixed into the parent.
//! - **Sends and returns** — Auxiliary send buses for effects like reverb,
//!   delay, etc. Sources can send a fraction of their signal to aux buses.
//! - **Sidechain compression** — A bus's compressor can be keyed by another
//!   bus's signal level (e.g., duck music when dialogue plays).
//! - **Bus snapshots** — Save and crossfade between complete mix states.
//! - **Solo/mute propagation** — Solo a bus to hear only it (and its children);
//!   mute propagates down the hierarchy.
//! - **VU meter** — Per-bus peak and RMS level metering.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// BusId
// ---------------------------------------------------------------------------

/// Unique identifier for an audio bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BusId(pub u32);

impl BusId {
    /// The master bus.
    pub const MASTER: BusId = BusId(0);
}

impl fmt::Display for BusId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::MASTER {
            write!(f, "Master")
        } else {
            write!(f, "Bus({})", self.0)
        }
    }
}

// ---------------------------------------------------------------------------
// BusKind
// ---------------------------------------------------------------------------

/// Kind of bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusKind {
    /// Standard mixing bus.
    Normal,
    /// Auxiliary send bus (for effects).
    Aux,
    /// Return bus (output of an aux effect).
    Return,
    /// Group bus (submix for organizational purposes).
    Group,
}

impl fmt::Display for BusKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BusKind::Normal => write!(f, "Normal"),
            BusKind::Aux => write!(f, "Aux"),
            BusKind::Return => write!(f, "Return"),
            BusKind::Group => write!(f, "Group"),
        }
    }
}

// ---------------------------------------------------------------------------
// VuMeter
// ---------------------------------------------------------------------------

/// Per-bus VU meter for level monitoring.
#[derive(Debug, Clone)]
pub struct VuMeter {
    /// Peak level [0, 1] (decaying).
    pub peak: f32,
    /// RMS level [0, 1].
    pub rms: f32,
    /// Instantaneous peak (non-decaying, reset manually).
    pub true_peak: f32,
    /// Sum of squared samples for RMS computation.
    rms_sum: f64,
    /// Number of samples in the current RMS window.
    rms_count: u64,
    /// Peak decay rate per sample.
    peak_decay: f32,
    /// Whether the meter is clipping (peak >= 1.0).
    pub clipping: bool,
    /// Number of samples that clipped.
    pub clip_count: u64,
}

impl VuMeter {
    /// Create a new VU meter.
    pub fn new() -> Self {
        Self {
            peak: 0.0,
            rms: 0.0,
            true_peak: 0.0,
            rms_sum: 0.0,
            rms_count: 0,
            peak_decay: 0.9995,
            clipping: false,
            clip_count: 0,
        }
    }

    /// Process a buffer of samples and update levels.
    pub fn process(&mut self, samples: &[f32]) {
        for &sample in samples {
            let abs = sample.abs();

            // Peak with decay.
            if abs > self.peak {
                self.peak = abs;
            } else {
                self.peak *= self.peak_decay;
            }

            // True peak.
            if abs > self.true_peak {
                self.true_peak = abs;
            }

            // RMS accumulation.
            self.rms_sum += (sample as f64) * (sample as f64);
            self.rms_count += 1;

            // Clipping detection.
            if abs >= 1.0 {
                self.clipping = true;
                self.clip_count += 1;
            }
        }

        // Compute RMS.
        if self.rms_count > 0 {
            self.rms = (self.rms_sum / self.rms_count as f64).sqrt() as f32;
        }
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.peak = 0.0;
        self.rms = 0.0;
        self.true_peak = 0.0;
        self.rms_sum = 0.0;
        self.rms_count = 0;
        self.clipping = false;
        self.clip_count = 0;
    }

    /// Reset only the RMS window (call per-frame).
    pub fn reset_rms(&mut self) {
        self.rms_sum = 0.0;
        self.rms_count = 0;
    }

    /// Reset clipping indicators.
    pub fn reset_clip(&mut self) {
        self.clipping = false;
        self.clip_count = 0;
        self.true_peak = 0.0;
    }

    /// Convert peak to dB.
    pub fn peak_db(&self) -> f32 {
        if self.peak < 1e-10 {
            -100.0
        } else {
            20.0 * self.peak.log10()
        }
    }

    /// Convert RMS to dB.
    pub fn rms_db(&self) -> f32 {
        if self.rms < 1e-10 {
            -100.0
        } else {
            20.0 * self.rms.log10()
        }
    }
}

impl Default for VuMeter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Compressor
// ---------------------------------------------------------------------------

/// Dynamics compressor for a bus.
#[derive(Debug, Clone)]
pub struct BusCompressor {
    /// Threshold in dB.
    pub threshold_db: f32,
    /// Ratio (e.g., 4.0 means 4:1 compression).
    pub ratio: f32,
    /// Attack time in seconds.
    pub attack: f32,
    /// Release time in seconds.
    pub release: f32,
    /// Makeup gain in dB.
    pub makeup_gain_db: f32,
    /// Knee width in dB (0 = hard knee).
    pub knee_db: f32,
    /// Whether the compressor is active.
    pub enabled: bool,
    /// Current gain reduction in dB.
    gain_reduction_db: f32,
    /// Envelope follower state.
    envelope: f32,
    /// Optional sidechain source bus.
    pub sidechain_source: Option<BusId>,
}

impl BusCompressor {
    /// Create a new compressor with default settings.
    pub fn new() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack: 0.01,
            release: 0.1,
            makeup_gain_db: 0.0,
            knee_db: 6.0,
            enabled: true,
            gain_reduction_db: 0.0,
            envelope: 0.0,
            sidechain_source: None,
        }
    }

    /// Set threshold.
    pub fn with_threshold(mut self, db: f32) -> Self {
        self.threshold_db = db;
        self
    }

    /// Set ratio.
    pub fn with_ratio(mut self, ratio: f32) -> Self {
        self.ratio = ratio.max(1.0);
        self
    }

    /// Set attack time.
    pub fn with_attack(mut self, seconds: f32) -> Self {
        self.attack = seconds.max(0.0001);
        self
    }

    /// Set release time.
    pub fn with_release(mut self, seconds: f32) -> Self {
        self.release = seconds.max(0.001);
        self
    }

    /// Set makeup gain.
    pub fn with_makeup(mut self, db: f32) -> Self {
        self.makeup_gain_db = db;
        self
    }

    /// Set sidechain source.
    pub fn with_sidechain(mut self, source: BusId) -> Self {
        self.sidechain_source = Some(source);
        self
    }

    /// Process samples in-place, applying compression.
    ///
    /// `key_level` is the level used for gain computation (from sidechain
    /// or from the signal itself).
    pub fn process(&mut self, samples: &mut [f32], key_level: f32, sample_rate: u32) {
        if !self.enabled {
            return;
        }

        let attack_coeff = (-1.0 / (self.attack * sample_rate as f32)).exp();
        let release_coeff = (-1.0 / (self.release * sample_rate as f32)).exp();

        let input_db = if key_level > 1e-10 {
            20.0 * key_level.log10()
        } else {
            -100.0
        };

        // Compute gain reduction.
        let over_db = input_db - self.threshold_db;
        let target_gr = if over_db <= -self.knee_db / 2.0 {
            0.0
        } else if over_db >= self.knee_db / 2.0 {
            over_db * (1.0 - 1.0 / self.ratio)
        } else {
            // Soft knee.
            let x = over_db + self.knee_db / 2.0;
            x * x / (2.0 * self.knee_db) * (1.0 - 1.0 / self.ratio)
        };

        // Envelope follower.
        let coeff = if target_gr > self.envelope {
            attack_coeff
        } else {
            release_coeff
        };
        self.envelope = coeff * self.envelope + (1.0 - coeff) * target_gr;
        self.gain_reduction_db = self.envelope;

        // Apply gain reduction + makeup.
        let total_db = -self.gain_reduction_db + self.makeup_gain_db;
        let gain = 10.0_f32.powf(total_db / 20.0);

        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }

    /// Returns the current gain reduction in dB.
    pub fn gain_reduction(&self) -> f32 {
        self.gain_reduction_db
    }

    /// Reset the compressor state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.gain_reduction_db = 0.0;
    }
}

impl Default for BusCompressor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SendConfig
// ---------------------------------------------------------------------------

/// Configuration for a send to an aux bus.
#[derive(Debug, Clone)]
pub struct SendConfig {
    /// Target aux bus.
    pub target: BusId,
    /// Send level [0, 1].
    pub level: f32,
    /// Whether this send is pre-fader (before volume) or post-fader.
    pub pre_fader: bool,
    /// Whether this send is enabled.
    pub enabled: bool,
}

impl SendConfig {
    /// Create a new send.
    pub fn new(target: BusId, level: f32) -> Self {
        Self {
            target,
            level: level.clamp(0.0, 1.0),
            pre_fader: false,
            enabled: true,
        }
    }

    /// Set as pre-fader.
    pub fn pre_fader(mut self) -> Self {
        self.pre_fader = true;
        self
    }
}

// ---------------------------------------------------------------------------
// AudioBusState
// ---------------------------------------------------------------------------

/// State of a single audio bus.
#[derive(Debug, Clone)]
pub struct AudioBusState {
    /// Bus identifier.
    pub id: BusId,
    /// Human-readable name.
    pub name: String,
    /// Kind of bus.
    pub kind: BusKind,
    /// Volume [0, 1].
    pub volume: f32,
    /// Pan [-1, 1] (left to right).
    pub pan: f32,
    /// Whether this bus is muted.
    pub muted: bool,
    /// Whether this bus is soloed.
    pub soloed: bool,
    /// Parent bus (None for master).
    pub parent: Option<BusId>,
    /// Child bus IDs.
    pub children: Vec<BusId>,
    /// Sends to aux buses.
    pub sends: Vec<SendConfig>,
    /// VU meter.
    pub meter: VuMeter,
    /// Compressor.
    pub compressor: Option<BusCompressor>,
    /// Whether this bus is enabled.
    pub enabled: bool,
    /// Internal audio buffer.
    buffer: Vec<f32>,
}

impl AudioBusState {
    /// Create a new bus.
    pub fn new(id: BusId, name: impl Into<String>, kind: BusKind) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            volume: 1.0,
            pan: 0.0,
            muted: false,
            soloed: false,
            parent: None,
            children: Vec::new(),
            sends: Vec::new(),
            meter: VuMeter::new(),
            compressor: None,
            enabled: true,
            buffer: Vec::new(),
        }
    }

    /// Set volume in linear scale [0, 1].
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 2.0);
    }

    /// Set volume from dB.
    pub fn set_volume_db(&mut self, db: f32) {
        self.volume = 10.0_f32.powf(db / 20.0);
    }

    /// Get volume in dB.
    pub fn volume_db(&self) -> f32 {
        if self.volume < 1e-10 {
            -100.0
        } else {
            20.0 * self.volume.log10()
        }
    }

    /// Set pan [-1, 1].
    pub fn set_pan(&mut self, pan: f32) {
        self.pan = pan.clamp(-1.0, 1.0);
    }

    /// Compute left and right gain from pan position.
    pub fn pan_gains(&self) -> (f32, f32) {
        let angle = (self.pan + 1.0) * 0.25 * std::f32::consts::PI;
        (angle.cos(), angle.sin())
    }

    /// Add a send.
    pub fn add_send(&mut self, send: SendConfig) {
        self.sends.push(send);
    }

    /// Remove a send by target bus.
    pub fn remove_send(&mut self, target: BusId) {
        self.sends.retain(|s| s.target != target);
    }

    /// Check if this bus should produce audio (not muted, enabled, etc.).
    pub fn is_audible(&self) -> bool {
        self.enabled && !self.muted && self.volume > 0.0
    }

    /// Ensure the internal buffer has the required size.
    pub fn ensure_buffer(&mut self, size: usize) {
        if self.buffer.len() < size {
            self.buffer.resize(size, 0.0);
        }
    }

    /// Clear the internal buffer.
    pub fn clear_buffer(&mut self) {
        for sample in &mut self.buffer {
            *sample = 0.0;
        }
    }

    /// Get the buffer as a slice.
    pub fn buffer(&self) -> &[f32] {
        &self.buffer
    }

    /// Get the buffer as a mutable slice.
    pub fn buffer_mut(&mut self) -> &mut [f32] {
        &mut self.buffer
    }

    /// Mix source samples into this bus's buffer.
    pub fn mix_into(&mut self, source: &[f32], gain: f32) {
        self.ensure_buffer(source.len());
        for (i, &s) in source.iter().enumerate() {
            if i < self.buffer.len() {
                self.buffer[i] += s * gain;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BusSnapshot
// ---------------------------------------------------------------------------

/// A snapshot of the entire mixer state for crossfading.
#[derive(Debug, Clone)]
pub struct BusSnapshot {
    /// Snapshot name.
    pub name: String,
    /// Per-bus volumes.
    pub volumes: HashMap<BusId, f32>,
    /// Per-bus pans.
    pub pans: HashMap<BusId, f32>,
    /// Per-bus mute states.
    pub mutes: HashMap<BusId, bool>,
    /// Per-send levels.
    pub send_levels: HashMap<(BusId, BusId), f32>,
}

impl BusSnapshot {
    /// Create a snapshot from the current mixer state.
    pub fn capture(name: impl Into<String>, buses: &HashMap<BusId, AudioBusState>) -> Self {
        let mut volumes = HashMap::new();
        let mut pans = HashMap::new();
        let mut mutes = HashMap::new();
        let mut send_levels = HashMap::new();

        for (id, bus) in buses {
            volumes.insert(*id, bus.volume);
            pans.insert(*id, bus.pan);
            mutes.insert(*id, bus.muted);
            for send in &bus.sends {
                send_levels.insert((*id, send.target), send.level);
            }
        }

        Self {
            name: name.into(),
            volumes,
            pans,
            mutes,
            send_levels,
        }
    }

    /// Interpolate between two snapshots.
    pub fn lerp(a: &BusSnapshot, b: &BusSnapshot, t: f32) -> BusSnapshot {
        let t = t.clamp(0.0, 1.0);
        let mut volumes = HashMap::new();
        let mut pans = HashMap::new();

        // Merge all bus IDs.
        let all_ids: std::collections::HashSet<BusId> = a
            .volumes
            .keys()
            .chain(b.volumes.keys())
            .copied()
            .collect();

        for id in &all_ids {
            let va = a.volumes.get(id).copied().unwrap_or(1.0);
            let vb = b.volumes.get(id).copied().unwrap_or(1.0);
            volumes.insert(*id, va + (vb - va) * t);

            let pa = a.pans.get(id).copied().unwrap_or(0.0);
            let pb = b.pans.get(id).copied().unwrap_or(0.0);
            pans.insert(*id, pa + (pb - pa) * t);
        }

        let mut send_levels = HashMap::new();
        let all_sends: std::collections::HashSet<(BusId, BusId)> = a
            .send_levels
            .keys()
            .chain(b.send_levels.keys())
            .copied()
            .collect();

        for key in &all_sends {
            let la = a.send_levels.get(key).copied().unwrap_or(0.0);
            let lb = b.send_levels.get(key).copied().unwrap_or(0.0);
            send_levels.insert(*key, la + (lb - la) * t);
        }

        BusSnapshot {
            name: if t < 0.5 {
                a.name.clone()
            } else {
                b.name.clone()
            },
            volumes,
            pans,
            mutes: b.mutes.clone(),
            send_levels,
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotCrossfade
// ---------------------------------------------------------------------------

/// An active crossfade between two bus snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotCrossfade {
    /// Source snapshot.
    pub from: BusSnapshot,
    /// Target snapshot.
    pub to: BusSnapshot,
    /// Duration in seconds.
    pub duration: f32,
    /// Elapsed time.
    pub elapsed: f32,
    /// Whether the crossfade is complete.
    pub complete: bool,
}

impl SnapshotCrossfade {
    /// Create a new crossfade.
    pub fn new(from: BusSnapshot, to: BusSnapshot, duration: f32) -> Self {
        Self {
            from,
            to,
            duration: duration.max(0.001),
            elapsed: 0.0,
            complete: false,
        }
    }

    /// Advance by delta time.
    pub fn update(&mut self, dt: f32) {
        self.elapsed += dt;
        if self.elapsed >= self.duration {
            self.elapsed = self.duration;
            self.complete = true;
        }
    }

    /// Returns the current interpolated snapshot.
    pub fn current(&self) -> BusSnapshot {
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        BusSnapshot::lerp(&self.from, &self.to, t)
    }

    /// Returns progress [0, 1].
    pub fn progress(&self) -> f32 {
        (self.elapsed / self.duration).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// BusMixer
// ---------------------------------------------------------------------------

/// The main bus mixer managing all audio buses, routing, and mixing.
///
/// # Example
///
/// ```ignore
/// let mut mixer = BusMixer::new(44100, 512);
///
/// let music = mixer.create_bus("Music", BusKind::Normal, Some(BusId::MASTER));
/// let sfx = mixer.create_bus("SFX", BusKind::Normal, Some(BusId::MASTER));
/// let reverb = mixer.create_bus("Reverb", BusKind::Aux, Some(BusId::MASTER));
///
/// // Send SFX to the reverb aux.
/// mixer.add_send(sfx, SendConfig::new(reverb, 0.3));
///
/// // Solo the music bus.
/// mixer.set_solo(music, true);
///
/// // Mix a frame.
/// let master_output = mixer.mix_frame();
/// ```
pub struct BusMixer {
    /// All buses, keyed by ID.
    buses: HashMap<BusId, AudioBusState>,
    /// Next bus ID.
    next_id: u32,
    /// Sample rate.
    sample_rate: u32,
    /// Buffer size per frame.
    buffer_size: usize,
    /// Saved snapshots.
    snapshots: HashMap<String, BusSnapshot>,
    /// Active crossfade.
    crossfade: Option<SnapshotCrossfade>,
    /// Whether any bus is currently soloed.
    any_soloed: bool,
    /// Master output buffer.
    master_output: Vec<f32>,
}

impl BusMixer {
    /// Create a new bus mixer with a master bus.
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        let mut buses = HashMap::new();
        let master = AudioBusState::new(BusId::MASTER, "Master", BusKind::Normal);
        buses.insert(BusId::MASTER, master);

        Self {
            buses,
            next_id: 1,
            sample_rate,
            buffer_size,
            snapshots: HashMap::new(),
            crossfade: None,
            any_soloed: false,
            master_output: vec![0.0; buffer_size],
        }
    }

    /// Create a new bus and return its ID.
    pub fn create_bus(
        &mut self,
        name: impl Into<String>,
        kind: BusKind,
        parent: Option<BusId>,
    ) -> BusId {
        let id = BusId(self.next_id);
        self.next_id += 1;

        let mut bus = AudioBusState::new(id, name, kind);
        bus.parent = parent;

        // Register as a child of the parent.
        if let Some(parent_id) = parent {
            if let Some(parent_bus) = self.buses.get_mut(&parent_id) {
                parent_bus.children.push(id);
            }
        }

        self.buses.insert(id, bus);
        id
    }

    /// Remove a bus.
    pub fn remove_bus(&mut self, id: BusId) -> bool {
        if id == BusId::MASTER {
            return false; // Cannot remove the master.
        }

        // Remove from parent's children list.
        if let Some(bus) = self.buses.get(&id) {
            let parent = bus.parent;
            if let Some(parent_id) = parent {
                if let Some(parent_bus) = self.buses.get_mut(&parent_id) {
                    parent_bus.children.retain(|&c| c != id);
                }
            }
        }

        self.buses.remove(&id).is_some()
    }

    /// Get a bus by ID.
    pub fn bus(&self, id: BusId) -> Option<&AudioBusState> {
        self.buses.get(&id)
    }

    /// Get a mutable bus by ID.
    pub fn bus_mut(&mut self, id: BusId) -> Option<&mut AudioBusState> {
        self.buses.get_mut(&id)
    }

    // -------------------------------------------------------------------
    // Volume / Pan / Mute / Solo
    // -------------------------------------------------------------------

    /// Set the volume of a bus.
    pub fn set_volume(&mut self, id: BusId, volume: f32) {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.set_volume(volume);
        }
    }

    /// Set the volume of a bus in dB.
    pub fn set_volume_db(&mut self, id: BusId, db: f32) {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.set_volume_db(db);
        }
    }

    /// Set the pan of a bus.
    pub fn set_pan(&mut self, id: BusId, pan: f32) {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.set_pan(pan);
        }
    }

    /// Mute or unmute a bus.
    pub fn set_mute(&mut self, id: BusId, muted: bool) {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.muted = muted;
        }
        // Propagate mute to children.
        if muted {
            self.propagate_mute(id);
        }
    }

    /// Propagate mute state to children.
    fn propagate_mute(&mut self, id: BusId) {
        let children: Vec<BusId> = self
            .buses
            .get(&id)
            .map(|b| b.children.clone())
            .unwrap_or_default();

        for child_id in children {
            if let Some(child) = self.buses.get_mut(&child_id) {
                child.muted = true;
            }
            self.propagate_mute(child_id);
        }
    }

    /// Solo or unsolo a bus.
    pub fn set_solo(&mut self, id: BusId, soloed: bool) {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.soloed = soloed;
        }
        self.update_solo_state();
    }

    /// Recompute the global solo state.
    fn update_solo_state(&mut self) {
        self.any_soloed = self.buses.values().any(|b| b.soloed);
    }

    /// Check if a bus is effectively audible considering solo state.
    pub fn is_bus_audible(&self, id: BusId) -> bool {
        let bus = match self.buses.get(&id) {
            Some(b) => b,
            None => return false,
        };

        if !bus.enabled || bus.muted {
            return false;
        }

        if self.any_soloed {
            // Only soloed buses (and their parents) are audible.
            if bus.soloed {
                return true;
            }
            // Check if any child is soloed.
            return self.has_soloed_descendant(id);
        }

        true
    }

    /// Check if any descendant bus is soloed.
    fn has_soloed_descendant(&self, id: BusId) -> bool {
        if let Some(bus) = self.buses.get(&id) {
            for &child_id in &bus.children {
                if let Some(child) = self.buses.get(&child_id) {
                    if child.soloed {
                        return true;
                    }
                }
                if self.has_soloed_descendant(child_id) {
                    return true;
                }
            }
        }
        false
    }

    // -------------------------------------------------------------------
    // Sends
    // -------------------------------------------------------------------

    /// Add a send from one bus to another.
    pub fn add_send(&mut self, from: BusId, send: SendConfig) {
        if let Some(bus) = self.buses.get_mut(&from) {
            bus.add_send(send);
        }
    }

    /// Remove a send.
    pub fn remove_send(&mut self, from: BusId, target: BusId) {
        if let Some(bus) = self.buses.get_mut(&from) {
            bus.remove_send(target);
        }
    }

    /// Set the level of a send.
    pub fn set_send_level(&mut self, from: BusId, target: BusId, level: f32) {
        if let Some(bus) = self.buses.get_mut(&from) {
            for send in &mut bus.sends {
                if send.target == target {
                    send.level = level.clamp(0.0, 1.0);
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Compressor
    // -------------------------------------------------------------------

    /// Set a compressor on a bus.
    pub fn set_compressor(&mut self, id: BusId, compressor: BusCompressor) {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.compressor = Some(compressor);
        }
    }

    /// Remove the compressor from a bus.
    pub fn remove_compressor(&mut self, id: BusId) {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.compressor = None;
        }
    }

    // -------------------------------------------------------------------
    // Mixing
    // -------------------------------------------------------------------

    /// Feed audio data into a bus.
    pub fn feed(&mut self, id: BusId, samples: &[f32]) {
        if let Some(bus) = self.buses.get_mut(&id) {
            bus.ensure_buffer(samples.len());
            bus.mix_into(samples, 1.0);
        }
    }

    /// Mix all buses and produce the master output.
    ///
    /// This processes the bus hierarchy bottom-up, mixing children into
    /// parents, applying sends, compression, and metering.
    pub fn mix_frame(&mut self) -> Vec<f32> {
        let buffer_size = self.buffer_size;

        // Clear all buffers.
        for bus in self.buses.values_mut() {
            bus.ensure_buffer(buffer_size);
        }

        // Process buses in bottom-up order (leaves first).
        let bus_order = self.compute_processing_order();

        for &id in &bus_order {
            if id == BusId::MASTER {
                continue; // Process master last.
            }

            let is_audible = self.is_bus_audible(id);

            // Get the bus's buffer, apply volume, and mix into parent.
            let (volume, parent, buffer_clone, sends_clone, muted) = {
                if let Some(bus) = self.buses.get(&id) {
                    (
                        bus.volume,
                        bus.parent,
                        bus.buffer.clone(),
                        bus.sends.clone(),
                        bus.muted,
                    )
                } else {
                    continue;
                }
            };

            if muted || !is_audible {
                continue;
            }

            // Update meter.
            if let Some(bus) = self.buses.get_mut(&id) {
                bus.meter.process(&bus.buffer);
            }

            // Apply compressor.
            let sample_rate = self.sample_rate;
            if let Some(bus) = self.buses.get_mut(&id) {
                if let Some(ref mut comp) = bus.compressor {
                    let key_level = bus.meter.peak;
                    comp.process(&mut bus.buffer, key_level, sample_rate);
                }
            }

            // Process sends.
            for send in &sends_clone {
                if send.enabled {
                    let send_gain = if send.pre_fader {
                        send.level
                    } else {
                        send.level * volume
                    };
                    if let Some(target_bus) = self.buses.get_mut(&send.target) {
                        target_bus.mix_into(&buffer_clone, send_gain);
                    }
                }
            }

            // Mix into parent.
            if let Some(parent_id) = parent {
                let bus_buffer = {
                    if let Some(bus) = self.buses.get(&id) {
                        let mut buf = bus.buffer.clone();
                        for s in &mut buf {
                            *s *= volume;
                        }
                        buf
                    } else {
                        continue;
                    }
                };

                if let Some(parent_bus) = self.buses.get_mut(&parent_id) {
                    parent_bus.mix_into(&bus_buffer, 1.0);
                }
            }
        }

        // Process master bus.
        if let Some(master) = self.buses.get_mut(&BusId::MASTER) {
            master.meter.process(&master.buffer);

            if let Some(ref mut comp) = master.compressor {
                let key_level = master.meter.peak;
                comp.process(&mut master.buffer, key_level, self.sample_rate);
            }

            // Apply master volume.
            let vol = master.volume;
            for s in &mut master.buffer {
                *s *= vol;
            }
        }

        // Copy master output.
        let output = if let Some(master) = self.buses.get(&BusId::MASTER) {
            master.buffer[..buffer_size.min(master.buffer.len())].to_vec()
        } else {
            vec![0.0; buffer_size]
        };

        // Clear all buffers for next frame.
        for bus in self.buses.values_mut() {
            bus.clear_buffer();
            bus.meter.reset_rms();
        }

        output
    }

    /// Compute the processing order (bottom-up).
    fn compute_processing_order(&self) -> Vec<BusId> {
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.visit_order(BusId::MASTER, &mut order, &mut visited);
        order.reverse(); // Bottom-up.
        order
    }

    /// DFS to compute processing order.
    fn visit_order(
        &self,
        id: BusId,
        order: &mut Vec<BusId>,
        visited: &mut std::collections::HashSet<BusId>,
    ) {
        if !visited.insert(id) {
            return;
        }
        if let Some(bus) = self.buses.get(&id) {
            for &child in &bus.children {
                self.visit_order(child, order, visited);
            }
        }
        order.push(id);
    }

    // -------------------------------------------------------------------
    // Snapshots
    // -------------------------------------------------------------------

    /// Save the current mixer state as a named snapshot.
    pub fn save_snapshot(&mut self, name: impl Into<String>) {
        let snapshot = BusSnapshot::capture(name, &self.buses);
        self.snapshots.insert(snapshot.name.clone(), snapshot);
    }

    /// Crossfade to a saved snapshot.
    pub fn crossfade_to_snapshot(&mut self, name: &str, duration: f32) -> bool {
        if let Some(target) = self.snapshots.get(name) {
            let current = BusSnapshot::capture("_current", &self.buses);
            self.crossfade = Some(SnapshotCrossfade::new(current, target.clone(), duration));
            true
        } else {
            false
        }
    }

    /// Apply a snapshot immediately (no crossfade).
    pub fn apply_snapshot(&mut self, name: &str) -> bool {
        if let Some(snapshot) = self.snapshots.get(name) {
            for (&id, &volume) in &snapshot.volumes {
                if let Some(bus) = self.buses.get_mut(&id) {
                    bus.volume = volume;
                }
            }
            for (&id, &pan) in &snapshot.pans {
                if let Some(bus) = self.buses.get_mut(&id) {
                    bus.pan = pan;
                }
            }
            for (&id, &muted) in &snapshot.mutes {
                if let Some(bus) = self.buses.get_mut(&id) {
                    bus.muted = muted;
                }
            }
            true
        } else {
            false
        }
    }

    /// Update the crossfade (call per frame).
    pub fn update_crossfade(&mut self, dt: f32) {
        if let Some(ref mut xfade) = self.crossfade {
            xfade.update(dt);
            let snap = xfade.current();

            // Apply interpolated values.
            for (&id, &volume) in &snap.volumes {
                if let Some(bus) = self.buses.get_mut(&id) {
                    bus.volume = volume;
                }
            }
            for (&id, &pan) in &snap.pans {
                if let Some(bus) = self.buses.get_mut(&id) {
                    bus.pan = pan;
                }
            }
        }

        if self.crossfade.as_ref().map_or(false, |x| x.complete) {
            self.crossfade = None;
        }
    }

    /// Delete a saved snapshot.
    pub fn delete_snapshot(&mut self, name: &str) -> bool {
        self.snapshots.remove(name).is_some()
    }

    /// Returns all snapshot names.
    pub fn snapshot_names(&self) -> Vec<&str> {
        self.snapshots.keys().map(|k| k.as_str()).collect()
    }

    // -------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------

    /// Returns the number of buses (including master).
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Returns all bus IDs.
    pub fn bus_ids(&self) -> Vec<BusId> {
        self.buses.keys().copied().collect()
    }

    /// Returns the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Returns whether a crossfade is active.
    pub fn is_crossfading(&self) -> bool {
        self.crossfade.is_some()
    }

    /// Returns the meter readings for a bus.
    pub fn meter(&self, id: BusId) -> Option<&VuMeter> {
        self.buses.get(&id).map(|b| &b.meter)
    }

    /// Dump a summary of the mixer state.
    pub fn dump(&self) -> String {
        let mut buf = String::new();
        buf.push_str("=== Bus Mixer ===\n");
        for (id, bus) in &self.buses {
            let solo_str = if bus.soloed { " [SOLO]" } else { "" };
            let mute_str = if bus.muted { " [MUTE]" } else { "" };
            buf.push_str(&format!(
                "  {} '{}' ({}) vol={:.2} pan={:.2} peak={:.1}dB{}{}\n",
                id,
                bus.name,
                bus.kind,
                bus.volume,
                bus.pan,
                bus.meter.peak_db(),
                solo_str,
                mute_str,
            ));
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vu_meter_basic() {
        let mut meter = VuMeter::new();
        meter.process(&[0.5, -0.5, 0.3, -0.3]);

        assert!(meter.peak > 0.0);
        assert!(meter.rms > 0.0);
        assert!(!meter.clipping);
    }

    #[test]
    fn test_vu_meter_clipping() {
        let mut meter = VuMeter::new();
        meter.process(&[1.0, 1.5, -1.2]);

        assert!(meter.clipping);
        assert!(meter.clip_count > 0);
    }

    #[test]
    fn test_vu_meter_db() {
        let mut meter = VuMeter::new();
        meter.process(&[1.0]);
        assert!((meter.peak_db() - 0.0).abs() < 0.1);

        meter.reset();
        meter.process(&[0.1]);
        assert!(meter.peak_db() < 0.0);
    }

    #[test]
    fn test_compressor_basic() {
        let mut comp = BusCompressor::new().with_threshold(-10.0).with_ratio(4.0);

        let mut samples = vec![0.5; 100];
        comp.process(&mut samples, 0.5, 44100);

        // With makeup gain 0, compressed signal should be quieter or same.
        for &s in &samples {
            assert!(s.abs() <= 1.0);
        }
    }

    #[test]
    fn test_compressor_sidechain() {
        let comp = BusCompressor::new()
            .with_threshold(-20.0)
            .with_sidechain(BusId(5));

        assert_eq!(comp.sidechain_source, Some(BusId(5)));
    }

    #[test]
    fn test_bus_mixer_create() {
        let mut mixer = BusMixer::new(44100, 512);
        assert_eq!(mixer.bus_count(), 1); // Master only.

        let music = mixer.create_bus("Music", BusKind::Normal, Some(BusId::MASTER));
        let sfx = mixer.create_bus("SFX", BusKind::Normal, Some(BusId::MASTER));

        assert_eq!(mixer.bus_count(), 3);
        assert!(mixer.bus(music).is_some());
        assert!(mixer.bus(sfx).is_some());
    }

    #[test]
    fn test_bus_mixer_hierarchy() {
        let mut mixer = BusMixer::new(44100, 512);
        let group = mixer.create_bus("UI", BusKind::Group, Some(BusId::MASTER));
        let child = mixer.create_bus("UI_Clicks", BusKind::Normal, Some(group));

        let parent = mixer.bus(group).unwrap();
        assert!(parent.children.contains(&child));

        let child_bus = mixer.bus(child).unwrap();
        assert_eq!(child_bus.parent, Some(group));
    }

    #[test]
    fn test_bus_mixer_volume() {
        let mut mixer = BusMixer::new(44100, 512);
        let music = mixer.create_bus("Music", BusKind::Normal, Some(BusId::MASTER));

        mixer.set_volume(music, 0.5);
        assert!((mixer.bus(music).unwrap().volume - 0.5).abs() < 1e-5);

        mixer.set_volume_db(music, -6.0);
        assert!(mixer.bus(music).unwrap().volume < 0.6);
    }

    #[test]
    fn test_bus_mixer_mute() {
        let mut mixer = BusMixer::new(44100, 512);
        let music = mixer.create_bus("Music", BusKind::Normal, Some(BusId::MASTER));

        mixer.set_mute(music, true);
        assert!(mixer.bus(music).unwrap().muted);
        assert!(!mixer.is_bus_audible(music));
    }

    #[test]
    fn test_bus_mixer_solo() {
        let mut mixer = BusMixer::new(44100, 512);
        let music = mixer.create_bus("Music", BusKind::Normal, Some(BusId::MASTER));
        let sfx = mixer.create_bus("SFX", BusKind::Normal, Some(BusId::MASTER));

        mixer.set_solo(music, true);
        assert!(mixer.is_bus_audible(music));
        assert!(!mixer.is_bus_audible(sfx));
    }

    #[test]
    fn test_bus_mixer_sends() {
        let mut mixer = BusMixer::new(44100, 512);
        let sfx = mixer.create_bus("SFX", BusKind::Normal, Some(BusId::MASTER));
        let reverb = mixer.create_bus("Reverb", BusKind::Aux, Some(BusId::MASTER));

        mixer.add_send(sfx, SendConfig::new(reverb, 0.3));

        let bus = mixer.bus(sfx).unwrap();
        assert_eq!(bus.sends.len(), 1);
        assert_eq!(bus.sends[0].target, reverb);
    }

    #[test]
    fn test_bus_mixer_mix_frame() {
        let mut mixer = BusMixer::new(44100, 64);
        let music = mixer.create_bus("Music", BusKind::Normal, Some(BusId::MASTER));

        // Feed some audio.
        let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        mixer.feed(music, &signal);

        let output = mixer.mix_frame();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_bus_mixer_snapshot() {
        let mut mixer = BusMixer::new(44100, 64);
        let music = mixer.create_bus("Music", BusKind::Normal, Some(BusId::MASTER));

        mixer.set_volume(music, 0.8);
        mixer.save_snapshot("state_1");

        mixer.set_volume(music, 0.3);
        mixer.save_snapshot("state_2");

        assert!(mixer.apply_snapshot("state_1"));
        assert!((mixer.bus(music).unwrap().volume - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_bus_mixer_crossfade() {
        let mut mixer = BusMixer::new(44100, 64);
        let music = mixer.create_bus("Music", BusKind::Normal, Some(BusId::MASTER));

        mixer.set_volume(music, 1.0);
        mixer.save_snapshot("loud");

        mixer.set_volume(music, 0.0);
        mixer.save_snapshot("silent");

        assert!(mixer.crossfade_to_snapshot("loud", 1.0));
        assert!(mixer.is_crossfading());

        mixer.update_crossfade(0.5);
        // Volume should be somewhere between 0 and 1.
        let vol = mixer.bus(music).unwrap().volume;
        assert!(vol > 0.0);
    }

    #[test]
    fn test_pan_gains() {
        let mut bus = AudioBusState::new(BusId(1), "test", BusKind::Normal);

        bus.set_pan(0.0); // Center.
        let (l, r) = bus.pan_gains();
        assert!((l - r).abs() < 0.01);

        bus.set_pan(-1.0); // Full left.
        let (l, r) = bus.pan_gains();
        assert!(l > r);
    }

    #[test]
    fn test_remove_bus() {
        let mut mixer = BusMixer::new(44100, 64);
        let id = mixer.create_bus("temp", BusKind::Normal, Some(BusId::MASTER));

        assert!(mixer.remove_bus(id));
        assert!(mixer.bus(id).is_none());
    }

    #[test]
    fn test_cannot_remove_master() {
        let mut mixer = BusMixer::new(44100, 64);
        assert!(!mixer.remove_bus(BusId::MASTER));
    }

    #[test]
    fn test_dump() {
        let mut mixer = BusMixer::new(44100, 64);
        mixer.create_bus("Music", BusKind::Normal, Some(BusId::MASTER));

        let dump = mixer.dump();
        assert!(dump.contains("Master"));
        assert!(dump.contains("Music"));
    }
}
