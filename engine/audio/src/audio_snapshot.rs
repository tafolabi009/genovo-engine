// engine/audio/src/audio_snapshot.rs
//
// Audio state snapshots for the Genovo engine.
//
// Capture and restore entire audio system states:
//
// - **Capture state** -- Save bus volumes, effect parameters, playing sounds.
// - **Restore snapshot** -- Apply a saved state instantly.
// - **Interpolate** -- Smoothly blend between two snapshots over time.
// - **Preset management** -- Named presets for common audio configurations.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_TRANSITION_DURATION: f32 = 1.0;
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SnapshotId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BusId(pub u32);

impl fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Snapshot({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Bus state
// ---------------------------------------------------------------------------

/// Captured state of a single audio bus.
#[derive(Debug, Clone)]
pub struct BusSnapshot {
    /// Bus identifier.
    pub bus_id: BusId,
    /// Bus name.
    pub name: String,
    /// Volume (0..1).
    pub volume: f32,
    /// Muted.
    pub muted: bool,
    /// Solo.
    pub solo: bool,
    /// Pan (-1 = left, 0 = center, 1 = right).
    pub pan: f32,
    /// Effect parameters per effect slot.
    pub effects: Vec<EffectSnapshot>,
    /// Send levels to other buses.
    pub sends: HashMap<BusId, f32>,
}

impl BusSnapshot {
    pub fn new(bus_id: BusId, name: &str) -> Self {
        Self {
            bus_id,
            name: name.to_string(),
            volume: 1.0,
            muted: false,
            solo: false,
            pan: 0.0,
            effects: Vec::new(),
            sends: HashMap::new(),
        }
    }

    /// Lerp between this bus state and another.
    pub fn lerp(&self, other: &BusSnapshot, t: f32) -> BusSnapshot {
        let t = t.clamp(0.0, 1.0);
        let mut result = self.clone();
        result.volume = self.volume + (other.volume - self.volume) * t;
        result.pan = self.pan + (other.pan - self.pan) * t;
        result.muted = if t < 0.5 { self.muted } else { other.muted };

        // Lerp sends.
        for (&bus, &level) in &other.sends {
            let current = self.sends.get(&bus).copied().unwrap_or(0.0);
            result.sends.insert(bus, current + (level - current) * t);
        }

        // Lerp effects.
        result.effects.clear();
        for (i, effect) in other.effects.iter().enumerate() {
            if let Some(self_effect) = self.effects.get(i) {
                result.effects.push(self_effect.lerp(effect, t));
            } else {
                result.effects.push(effect.clone());
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Effect snapshot
// ---------------------------------------------------------------------------

/// Captured state of a single audio effect.
#[derive(Debug, Clone)]
pub struct EffectSnapshot {
    /// Effect type name.
    pub effect_type: String,
    /// Effect parameters.
    pub params: HashMap<String, f32>,
    /// Whether the effect is enabled.
    pub enabled: bool,
    /// Wet/dry mix (0 = fully dry, 1 = fully wet).
    pub mix: f32,
}

impl EffectSnapshot {
    pub fn new(effect_type: &str) -> Self {
        Self {
            effect_type: effect_type.to_string(),
            params: HashMap::new(),
            enabled: true,
            mix: 1.0,
        }
    }

    pub fn with_param(mut self, name: &str, value: f32) -> Self {
        self.params.insert(name.to_string(), value);
        self
    }

    /// Lerp between two effect states.
    pub fn lerp(&self, other: &EffectSnapshot, t: f32) -> EffectSnapshot {
        let mut result = self.clone();
        result.mix = self.mix + (other.mix - self.mix) * t;
        result.enabled = if t < 0.5 { self.enabled } else { other.enabled };

        for (key, &other_val) in &other.params {
            let self_val = self.params.get(key).copied().unwrap_or(0.0);
            result.params.insert(key.clone(), self_val + (other_val - self_val) * t);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Sound snapshot
// ---------------------------------------------------------------------------

/// Captured state of a playing sound.
#[derive(Debug, Clone)]
pub struct SoundSnapshot {
    /// Sound asset identifier.
    pub asset: String,
    /// Volume.
    pub volume: f32,
    /// Pitch multiplier.
    pub pitch: f32,
    /// Whether the sound is playing.
    pub playing: bool,
    /// Playback position (seconds).
    pub position: f32,
    /// Whether the sound is looping.
    pub looping: bool,
    /// Bus assignment.
    pub bus: BusId,
    /// 3D position (None = 2D sound).
    pub world_position: Option<[f32; 3]>,
}

// ---------------------------------------------------------------------------
// Audio snapshot
// ---------------------------------------------------------------------------

/// A complete audio state snapshot.
#[derive(Debug, Clone)]
pub struct AudioStateSnapshot {
    /// Snapshot identifier.
    pub id: SnapshotId,
    /// Name.
    pub name: String,
    /// Bus states.
    pub buses: Vec<BusSnapshot>,
    /// Global master volume.
    pub master_volume: f32,
    /// Listener position.
    pub listener_position: [f32; 3],
    /// Listener forward direction.
    pub listener_forward: [f32; 3],
    /// Active sounds (optional, for full state save).
    pub sounds: Vec<SoundSnapshot>,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
    /// Creation timestamp.
    pub timestamp: f64,
}

impl AudioStateSnapshot {
    /// Create a new empty snapshot.
    pub fn new(id: SnapshotId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            buses: Vec::new(),
            master_volume: 1.0,
            listener_position: [0.0; 3],
            listener_forward: [0.0, 0.0, -1.0],
            sounds: Vec::new(),
            metadata: HashMap::new(),
            timestamp: 0.0,
        }
    }

    /// Add a bus state.
    pub fn add_bus(&mut self, bus: BusSnapshot) {
        self.buses.push(bus);
    }

    /// Get a bus by ID.
    pub fn bus(&self, id: BusId) -> Option<&BusSnapshot> {
        self.buses.iter().find(|b| b.bus_id == id)
    }

    /// Lerp between two snapshots.
    pub fn lerp(a: &AudioStateSnapshot, b: &AudioStateSnapshot, t: f32) -> AudioStateSnapshot {
        let t = t.clamp(0.0, 1.0);
        let mut result = a.clone();
        result.master_volume = a.master_volume + (b.master_volume - a.master_volume) * t;

        for i in 0..3 {
            result.listener_position[i] =
                a.listener_position[i] + (b.listener_position[i] - a.listener_position[i]) * t;
            result.listener_forward[i] =
                a.listener_forward[i] + (b.listener_forward[i] - a.listener_forward[i]) * t;
        }

        // Lerp buses.
        result.buses.clear();
        for a_bus in &a.buses {
            if let Some(b_bus) = b.buses.iter().find(|bb| bb.bus_id == a_bus.bus_id) {
                result.buses.push(a_bus.lerp(b_bus, t));
            } else {
                result.buses.push(a_bus.clone());
            }
        }
        // Add buses only in b.
        for b_bus in &b.buses {
            if !a.buses.iter().any(|ab| ab.bus_id == b_bus.bus_id) {
                let mut new_bus = b_bus.clone();
                new_bus.volume *= t;
                result.buses.push(new_bus);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Transition state
// ---------------------------------------------------------------------------

/// A transition between two snapshots.
#[derive(Debug, Clone)]
struct SnapshotTransition {
    from: AudioStateSnapshot,
    to: AudioStateSnapshot,
    duration: f32,
    elapsed: f32,
    active: bool,
}

impl SnapshotTransition {
    fn progress(&self) -> f32 {
        if self.duration <= EPSILON {
            1.0
        } else {
            (self.elapsed / self.duration).clamp(0.0, 1.0)
        }
    }

    fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }

    fn current(&self) -> AudioStateSnapshot {
        AudioStateSnapshot::lerp(&self.from, &self.to, self.progress())
    }
}

// ---------------------------------------------------------------------------
// Preset
// ---------------------------------------------------------------------------

/// A named audio preset.
#[derive(Debug, Clone)]
pub struct AudioPreset {
    pub name: String,
    pub description: String,
    pub snapshot: AudioStateSnapshot,
    pub category: String,
}

impl AudioPreset {
    pub fn new(name: &str, snapshot: AudioStateSnapshot) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            snapshot,
            category: "General".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Snapshot manager
// ---------------------------------------------------------------------------

/// Events from the snapshot system.
#[derive(Debug, Clone)]
pub enum SnapshotEvent {
    SnapshotCaptured { id: SnapshotId, name: String },
    SnapshotRestored { id: SnapshotId },
    TransitionStarted { from: SnapshotId, to: SnapshotId, duration: f32 },
    TransitionCompleted { to: SnapshotId },
    PresetSaved { name: String },
    PresetLoaded { name: String },
}

/// Manages audio state snapshots and transitions.
pub struct AudioSnapshotManager {
    /// Stored snapshots.
    snapshots: HashMap<SnapshotId, AudioStateSnapshot>,
    /// Named presets.
    presets: HashMap<String, AudioPreset>,
    /// Current active snapshot.
    current: Option<AudioStateSnapshot>,
    /// Active transition.
    transition: Option<SnapshotTransition>,
    /// Next snapshot ID.
    next_id: u32,
    /// Events.
    events: Vec<SnapshotEvent>,
    /// Default transition duration.
    default_transition_duration: f32,
}

impl AudioSnapshotManager {
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            presets: HashMap::new(),
            current: None,
            transition: None,
            next_id: 0,
            events: Vec::new(),
            default_transition_duration: DEFAULT_TRANSITION_DURATION,
        }
    }

    /// Capture the current audio state.
    pub fn capture(&mut self, name: &str, state: AudioStateSnapshot) -> SnapshotId {
        let id = SnapshotId(self.next_id);
        self.next_id += 1;
        let mut snapshot = state;
        snapshot.id = id;
        snapshot.name = name.to_string();
        self.snapshots.insert(id, snapshot);
        self.events.push(SnapshotEvent::SnapshotCaptured {
            id,
            name: name.to_string(),
        });
        id
    }

    /// Get a stored snapshot.
    pub fn snapshot(&self, id: SnapshotId) -> Option<&AudioStateSnapshot> {
        self.snapshots.get(&id)
    }

    /// Restore a snapshot instantly.
    pub fn restore(&mut self, id: SnapshotId) -> bool {
        if let Some(snapshot) = self.snapshots.get(&id).cloned() {
            self.current = Some(snapshot);
            self.transition = None;
            self.events.push(SnapshotEvent::SnapshotRestored { id });
            true
        } else {
            false
        }
    }

    /// Transition to a snapshot over time.
    pub fn transition_to(&mut self, target_id: SnapshotId, duration: f32) -> bool {
        let target = match self.snapshots.get(&target_id).cloned() {
            Some(s) => s,
            None => return false,
        };

        let from = self
            .current
            .clone()
            .unwrap_or_else(|| AudioStateSnapshot::new(SnapshotId(0), "empty"));

        let from_id = from.id;

        self.transition = Some(SnapshotTransition {
            from,
            to: target,
            duration,
            elapsed: 0.0,
            active: true,
        });

        self.events.push(SnapshotEvent::TransitionStarted {
            from: from_id,
            to: target_id,
            duration,
        });

        true
    }

    /// Transition to a snapshot using the default duration.
    pub fn transition_to_default(&mut self, target_id: SnapshotId) -> bool {
        let duration = self.default_transition_duration;
        self.transition_to(target_id, duration)
    }

    /// Save the current state as a named preset.
    pub fn save_preset(&mut self, name: &str) {
        if let Some(current) = &self.current {
            let preset = AudioPreset::new(name, current.clone());
            self.presets.insert(name.to_string(), preset);
            self.events.push(SnapshotEvent::PresetSaved {
                name: name.to_string(),
            });
        }
    }

    /// Load a preset by name.
    pub fn load_preset(&mut self, name: &str, transition_duration: f32) -> bool {
        if let Some(preset) = self.presets.get(name).cloned() {
            let id = self.capture(&format!("preset:{}", name), preset.snapshot);
            self.events.push(SnapshotEvent::PresetLoaded {
                name: name.to_string(),
            });
            if transition_duration > EPSILON {
                self.transition_to(id, transition_duration)
            } else {
                self.restore(id)
            }
        } else {
            false
        }
    }

    /// Update transitions.
    pub fn update(&mut self, dt: f32) -> Option<AudioStateSnapshot> {
        if let Some(transition) = &mut self.transition {
            transition.elapsed += dt;
            let result = transition.current();

            if transition.is_complete() {
                let to_id = transition.to.id;
                self.current = Some(transition.to.clone());
                self.transition = None;
                self.events.push(SnapshotEvent::TransitionCompleted { to: to_id });
                return Some(self.current.clone().unwrap());
            }

            return Some(result);
        }

        self.current.clone()
    }

    /// Get the current interpolated state.
    pub fn current_state(&self) -> Option<&AudioStateSnapshot> {
        self.current.as_ref()
    }

    /// Whether a transition is in progress.
    pub fn is_transitioning(&self) -> bool {
        self.transition.is_some()
    }

    /// Get transition progress (0..1).
    pub fn transition_progress(&self) -> f32 {
        self.transition.as_ref().map(|t| t.progress()).unwrap_or(1.0)
    }

    /// Get preset names.
    pub fn preset_names(&self) -> Vec<&str> {
        self.presets.keys().map(|s| s.as_str()).collect()
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<SnapshotEvent> {
        std::mem::take(&mut self.events)
    }

    /// Set the default transition duration.
    pub fn set_default_transition(&mut self, duration: f32) {
        self.default_transition_duration = duration.max(0.0);
    }

    /// Get stored snapshot count.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get preset count.
    pub fn preset_count(&self) -> usize {
        self.presets.len()
    }
}

impl Default for AudioSnapshotManager {
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

    fn make_snapshot(vol: f32) -> AudioStateSnapshot {
        let mut snap = AudioStateSnapshot::new(SnapshotId(0), "test");
        snap.master_volume = vol;
        snap.add_bus(BusSnapshot::new(BusId(0), "Master"));
        snap
    }

    #[test]
    fn test_capture_restore() {
        let mut mgr = AudioSnapshotManager::new();
        let id = mgr.capture("test", make_snapshot(0.5));
        mgr.restore(id);
        assert!((mgr.current_state().unwrap().master_volume - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_transition() {
        let mut mgr = AudioSnapshotManager::new();
        let from_id = mgr.capture("from", make_snapshot(0.0));
        let to_id = mgr.capture("to", make_snapshot(1.0));
        mgr.restore(from_id);

        mgr.transition_to(to_id, 1.0);
        assert!(mgr.is_transitioning());

        let mid = mgr.update(0.5).unwrap();
        assert!(mid.master_volume > 0.2);
        assert!(mid.master_volume < 0.8);

        mgr.update(1.0); // Complete.
        assert!(!mgr.is_transitioning());
    }

    #[test]
    fn test_lerp_snapshot() {
        let a = make_snapshot(0.0);
        let b = make_snapshot(1.0);
        let mid = AudioStateSnapshot::lerp(&a, &b, 0.5);
        assert!((mid.master_volume - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_presets() {
        let mut mgr = AudioSnapshotManager::new();
        mgr.capture("base", make_snapshot(0.5));
        mgr.restore(SnapshotId(0));
        mgr.save_preset("default");
        assert!(mgr.preset_names().contains(&"default"));
    }
}
