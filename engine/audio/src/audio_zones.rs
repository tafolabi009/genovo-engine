//! # Audio Environment Zones
//!
//! Zone-based audio environment system for the Genovo engine. Defines
//! spatial regions with distinct acoustic properties (reverb, EQ) and
//! handles smooth transitions as the listener moves between zones.
//!
//! ## Features
//!
//! - **Reverb zones** — Per-zone reverb with room size, reflection, and
//!   diffusion parameters.
//! - **EQ zones** — Outdoor/indoor frequency balance presets.
//! - **Zone blending** — Smooth crossfade when transitioning between zones.
//! - **Priority system** — Overlapping zones resolve by priority.
//! - **Snapshot system** — Save and restore complete audio environment state.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ZoneId
// ---------------------------------------------------------------------------

/// Unique identifier for an audio zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZoneId(pub u32);

impl fmt::Display for ZoneId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zone({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// ZoneShape
// ---------------------------------------------------------------------------

/// Shape of an audio zone's spatial region.
#[derive(Debug, Clone)]
pub enum ZoneShape {
    /// Axis-aligned bounding box.
    Box {
        center: [f32; 3],
        half_extents: [f32; 3],
    },
    /// Sphere.
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    /// Cylinder (vertical axis).
    Cylinder {
        center: [f32; 3],
        radius: f32,
        half_height: f32,
    },
    /// Capsule (vertical axis).
    Capsule {
        center: [f32; 3],
        radius: f32,
        half_height: f32,
    },
}

impl ZoneShape {
    /// Check if a point is inside this shape.
    pub fn contains(&self, point: &[f32; 3]) -> bool {
        match self {
            ZoneShape::Box { center, half_extents } => {
                (point[0] - center[0]).abs() <= half_extents[0]
                    && (point[1] - center[1]).abs() <= half_extents[1]
                    && (point[2] - center[2]).abs() <= half_extents[2]
            }
            ZoneShape::Sphere { center, radius } => {
                let dx = point[0] - center[0];
                let dy = point[1] - center[1];
                let dz = point[2] - center[2];
                dx * dx + dy * dy + dz * dz <= radius * radius
            }
            ZoneShape::Cylinder { center, radius, half_height } => {
                let dx = point[0] - center[0];
                let dz = point[2] - center[2];
                let dy = (point[1] - center[1]).abs();
                dx * dx + dz * dz <= radius * radius && dy <= *half_height
            }
            ZoneShape::Capsule { center, radius, half_height } => {
                let dx = point[0] - center[0];
                let dz = point[2] - center[2];
                let dy = point[1] - center[1];
                let horizontal_dist_sq = dx * dx + dz * dz;

                if dy.abs() <= *half_height {
                    // In the cylinder part.
                    horizontal_dist_sq <= radius * radius
                } else {
                    // In the hemisphere caps.
                    let cap_center_y = if dy > 0.0 {
                        center[1] + half_height
                    } else {
                        center[1] - half_height
                    };
                    let cap_dy = point[1] - cap_center_y;
                    horizontal_dist_sq + cap_dy * cap_dy <= radius * radius
                }
            }
        }
    }

    /// Compute the distance from a point to the nearest surface of this shape.
    ///
    /// Negative = inside, positive = outside.
    pub fn signed_distance(&self, point: &[f32; 3]) -> f32 {
        match self {
            ZoneShape::Box { center, half_extents } => {
                let dx = (point[0] - center[0]).abs() - half_extents[0];
                let dy = (point[1] - center[1]).abs() - half_extents[1];
                let dz = (point[2] - center[2]).abs() - half_extents[2];
                let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt();
                let inside = dx.max(dy).max(dz).min(0.0);
                outside + inside
            }
            ZoneShape::Sphere { center, radius } => {
                let dx = point[0] - center[0];
                let dy = point[1] - center[1];
                let dz = point[2] - center[2];
                (dx * dx + dy * dy + dz * dz).sqrt() - radius
            }
            ZoneShape::Cylinder { center, radius, half_height } => {
                let dx = point[0] - center[0];
                let dz = point[2] - center[2];
                let horizontal = (dx * dx + dz * dz).sqrt() - radius;
                let vertical = (point[1] - center[1]).abs() - half_height;
                if horizontal > 0.0 && vertical > 0.0 {
                    (horizontal * horizontal + vertical * vertical).sqrt()
                } else {
                    horizontal.max(vertical)
                }
            }
            ZoneShape::Capsule { center, radius, half_height } => {
                let dx = point[0] - center[0];
                let dz = point[2] - center[2];
                let dy = point[1] - center[1];
                let clamped_y = dy.clamp(-half_height, *half_height);
                let dist_x = dx;
                let dist_y = dy - clamped_y;
                let dist_z = dz;
                (dist_x * dist_x + dist_y * dist_y + dist_z * dist_z).sqrt() - radius
            }
        }
    }

    /// Compute blend weight based on distance from the zone boundary.
    ///
    /// Returns 1.0 when fully inside, 0.0 when outside the blend radius.
    pub fn blend_weight(&self, point: &[f32; 3], blend_radius: f32) -> f32 {
        let sd = self.signed_distance(point);
        if sd <= -blend_radius {
            1.0
        } else if sd >= 0.0 {
            0.0
        } else {
            // Smooth blend in the transition region.
            let t = (-sd) / blend_radius;
            // Smoothstep for perceptually smooth transition.
            t * t * (3.0 - 2.0 * t)
        }
    }
}

// ---------------------------------------------------------------------------
// ReverbParams
// ---------------------------------------------------------------------------

/// Reverb parameters for a zone.
#[derive(Debug, Clone)]
pub struct ReverbParams {
    /// Room size factor [0, 1]. Affects late reflection density.
    pub room_size: f32,
    /// Damping [0, 1]. Higher values absorb high frequencies faster.
    pub damping: f32,
    /// Wet/dry mix [0, 1].
    pub wet: f32,
    /// Diffusion [0, 1]. How scattered the reflections are.
    pub diffusion: f32,
    /// Pre-delay in seconds.
    pub pre_delay: f32,
    /// Decay time (RT60) in seconds.
    pub decay_time: f32,
    /// High-frequency decay ratio [0, 1].
    pub hf_decay_ratio: f32,
    /// Low-frequency decay ratio [0, 1].
    pub lf_decay_ratio: f32,
    /// Early reflection level in dB.
    pub early_level_db: f32,
    /// Late reflection level in dB.
    pub late_level_db: f32,
}

impl ReverbParams {
    /// Create default reverb parameters.
    pub fn new() -> Self {
        Self {
            room_size: 0.5,
            damping: 0.5,
            wet: 0.3,
            diffusion: 0.7,
            pre_delay: 0.01,
            decay_time: 1.0,
            hf_decay_ratio: 0.5,
            lf_decay_ratio: 1.0,
            early_level_db: -3.0,
            late_level_db: -6.0,
        }
    }

    /// Small room preset.
    pub fn small_room() -> Self {
        Self {
            room_size: 0.2,
            damping: 0.7,
            wet: 0.2,
            diffusion: 0.5,
            pre_delay: 0.005,
            decay_time: 0.4,
            hf_decay_ratio: 0.4,
            lf_decay_ratio: 1.0,
            early_level_db: -2.0,
            late_level_db: -8.0,
        }
    }

    /// Large hall preset.
    pub fn large_hall() -> Self {
        Self {
            room_size: 0.9,
            damping: 0.3,
            wet: 0.4,
            diffusion: 0.8,
            pre_delay: 0.03,
            decay_time: 2.5,
            hf_decay_ratio: 0.6,
            lf_decay_ratio: 1.0,
            early_level_db: -4.0,
            late_level_db: -5.0,
        }
    }

    /// Cathedral preset.
    pub fn cathedral() -> Self {
        Self {
            room_size: 1.0,
            damping: 0.2,
            wet: 0.5,
            diffusion: 0.9,
            pre_delay: 0.05,
            decay_time: 4.0,
            hf_decay_ratio: 0.7,
            lf_decay_ratio: 1.0,
            early_level_db: -5.0,
            late_level_db: -3.0,
        }
    }

    /// Outdoor preset (minimal reverb).
    pub fn outdoor() -> Self {
        Self {
            room_size: 0.05,
            damping: 0.9,
            wet: 0.05,
            diffusion: 0.3,
            pre_delay: 0.0,
            decay_time: 0.2,
            hf_decay_ratio: 0.3,
            lf_decay_ratio: 0.8,
            early_level_db: -10.0,
            late_level_db: -20.0,
        }
    }

    /// Bathroom preset (bright, tight reflections).
    pub fn bathroom() -> Self {
        Self {
            room_size: 0.15,
            damping: 0.3,
            wet: 0.35,
            diffusion: 0.4,
            pre_delay: 0.003,
            decay_time: 0.8,
            hf_decay_ratio: 0.8,
            lf_decay_ratio: 1.0,
            early_level_db: -1.0,
            late_level_db: -5.0,
        }
    }

    /// Linear interpolation between two reverb parameter sets.
    pub fn lerp(a: &ReverbParams, b: &ReverbParams, t: f32) -> ReverbParams {
        let t = t.clamp(0.0, 1.0);
        ReverbParams {
            room_size: a.room_size + (b.room_size - a.room_size) * t,
            damping: a.damping + (b.damping - a.damping) * t,
            wet: a.wet + (b.wet - a.wet) * t,
            diffusion: a.diffusion + (b.diffusion - a.diffusion) * t,
            pre_delay: a.pre_delay + (b.pre_delay - a.pre_delay) * t,
            decay_time: a.decay_time + (b.decay_time - a.decay_time) * t,
            hf_decay_ratio: a.hf_decay_ratio + (b.hf_decay_ratio - a.hf_decay_ratio) * t,
            lf_decay_ratio: a.lf_decay_ratio + (b.lf_decay_ratio - a.lf_decay_ratio) * t,
            early_level_db: a.early_level_db + (b.early_level_db - a.early_level_db) * t,
            late_level_db: a.late_level_db + (b.late_level_db - a.late_level_db) * t,
        }
    }
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EqParams
// ---------------------------------------------------------------------------

/// Equalizer parameters for a zone.
#[derive(Debug, Clone)]
pub struct EqParams {
    /// Low band gain in dB (< 200 Hz).
    pub low_gain_db: f32,
    /// Low-mid band gain in dB (200-1000 Hz).
    pub low_mid_gain_db: f32,
    /// Mid band gain in dB (1-4 kHz).
    pub mid_gain_db: f32,
    /// High-mid band gain in dB (4-10 kHz).
    pub high_mid_gain_db: f32,
    /// High band gain in dB (> 10 kHz).
    pub high_gain_db: f32,
}

impl EqParams {
    /// Flat EQ (no adjustment).
    pub fn flat() -> Self {
        Self {
            low_gain_db: 0.0,
            low_mid_gain_db: 0.0,
            mid_gain_db: 0.0,
            high_mid_gain_db: 0.0,
            high_gain_db: 0.0,
        }
    }

    /// Indoor preset — slightly boosted lows, cut highs.
    pub fn indoor() -> Self {
        Self {
            low_gain_db: 2.0,
            low_mid_gain_db: 1.0,
            mid_gain_db: 0.0,
            high_mid_gain_db: -1.0,
            high_gain_db: -3.0,
        }
    }

    /// Outdoor preset — natural roll-off, boosted highs (open air).
    pub fn outdoor() -> Self {
        Self {
            low_gain_db: -2.0,
            low_mid_gain_db: 0.0,
            mid_gain_db: 0.0,
            high_mid_gain_db: 1.0,
            high_gain_db: 2.0,
        }
    }

    /// Cave/underground preset — heavy bass, cut highs.
    pub fn cave() -> Self {
        Self {
            low_gain_db: 4.0,
            low_mid_gain_db: 2.0,
            mid_gain_db: -1.0,
            high_mid_gain_db: -3.0,
            high_gain_db: -5.0,
        }
    }

    /// Underwater preset — extreme muffling.
    pub fn underwater() -> Self {
        Self {
            low_gain_db: 3.0,
            low_mid_gain_db: 0.0,
            mid_gain_db: -4.0,
            high_mid_gain_db: -8.0,
            high_gain_db: -12.0,
        }
    }

    /// Metal corridor preset.
    pub fn metal_corridor() -> Self {
        Self {
            low_gain_db: -1.0,
            low_mid_gain_db: 1.0,
            mid_gain_db: 3.0,
            high_mid_gain_db: 2.0,
            high_gain_db: 1.0,
        }
    }

    /// Lerp between two EQ parameter sets.
    pub fn lerp(a: &EqParams, b: &EqParams, t: f32) -> EqParams {
        let t = t.clamp(0.0, 1.0);
        EqParams {
            low_gain_db: a.low_gain_db + (b.low_gain_db - a.low_gain_db) * t,
            low_mid_gain_db: a.low_mid_gain_db + (b.low_mid_gain_db - a.low_mid_gain_db) * t,
            mid_gain_db: a.mid_gain_db + (b.mid_gain_db - a.mid_gain_db) * t,
            high_mid_gain_db: a.high_mid_gain_db + (b.high_mid_gain_db - a.high_mid_gain_db) * t,
            high_gain_db: a.high_gain_db + (b.high_gain_db - a.high_gain_db) * t,
        }
    }
}

impl Default for EqParams {
    fn default() -> Self {
        Self::flat()
    }
}

// ---------------------------------------------------------------------------
// AudioZone
// ---------------------------------------------------------------------------

/// An audio environment zone with reverb, EQ, and priority.
#[derive(Debug, Clone)]
pub struct AudioZone {
    /// Unique identifier.
    pub id: ZoneId,
    /// Human-readable name.
    pub name: String,
    /// Spatial shape.
    pub shape: ZoneShape,
    /// Reverb parameters.
    pub reverb: ReverbParams,
    /// EQ parameters.
    pub eq: EqParams,
    /// Priority for overlapping zones. Higher = takes precedence.
    pub priority: i32,
    /// Blend radius for smooth transitions (world units).
    pub blend_radius: f32,
    /// Volume adjustment in dB.
    pub volume_db: f32,
    /// Whether this zone is currently active.
    pub enabled: bool,
    /// Optional tag for grouping.
    pub tag: Option<String>,
}

impl AudioZone {
    /// Create a new audio zone.
    pub fn new(
        id: ZoneId,
        name: impl Into<String>,
        shape: ZoneShape,
        reverb: ReverbParams,
        eq: EqParams,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            shape,
            reverb,
            eq,
            priority: 0,
            blend_radius: 2.0,
            volume_db: 0.0,
            enabled: true,
            tag: None,
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the blend radius.
    pub fn with_blend_radius(mut self, radius: f32) -> Self {
        self.blend_radius = radius;
        self
    }

    /// Set the volume adjustment.
    pub fn with_volume_db(mut self, db: f32) -> Self {
        self.volume_db = db;
        self
    }

    /// Set a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// Compute the blend weight for a listener position.
    pub fn blend_weight(&self, listener_pos: &[f32; 3]) -> f32 {
        self.shape.blend_weight(listener_pos, self.blend_radius)
    }
}

// ---------------------------------------------------------------------------
// AudioSnapshot
// ---------------------------------------------------------------------------

/// A snapshot of the complete audio environment state.
///
/// Captures the active zone blend, reverb, EQ, and volume so it can be
/// restored later or crossfaded to.
#[derive(Debug, Clone)]
pub struct AudioSnapshot {
    /// Snapshot name.
    pub name: String,
    /// Blended reverb parameters at capture time.
    pub reverb: ReverbParams,
    /// Blended EQ parameters at capture time.
    pub eq: EqParams,
    /// Volume in dB.
    pub volume_db: f32,
    /// Per-zone weights at capture time.
    pub zone_weights: HashMap<ZoneId, f32>,
}

impl AudioSnapshot {
    /// Create a snapshot from current state.
    pub fn capture(
        name: impl Into<String>,
        reverb: &ReverbParams,
        eq: &EqParams,
        volume_db: f32,
        zone_weights: &HashMap<ZoneId, f32>,
    ) -> Self {
        Self {
            name: name.into(),
            reverb: reverb.clone(),
            eq: eq.clone(),
            volume_db,
            zone_weights: zone_weights.clone(),
        }
    }

    /// Linear interpolation between two snapshots.
    pub fn lerp(a: &AudioSnapshot, b: &AudioSnapshot, t: f32) -> AudioSnapshot {
        AudioSnapshot {
            name: if t < 0.5 {
                a.name.clone()
            } else {
                b.name.clone()
            },
            reverb: ReverbParams::lerp(&a.reverb, &b.reverb, t),
            eq: EqParams::lerp(&a.eq, &b.eq, t),
            volume_db: a.volume_db + (b.volume_db - a.volume_db) * t,
            zone_weights: b.zone_weights.clone(), // Use target weights.
        }
    }
}

// ---------------------------------------------------------------------------
// BlendState
// ---------------------------------------------------------------------------

/// Current blend state between zones.
#[derive(Debug, Clone)]
pub struct BlendState {
    /// Per-zone blend weights (after priority resolution).
    pub zone_weights: HashMap<ZoneId, f32>,
    /// Blended reverb parameters.
    pub reverb: ReverbParams,
    /// Blended EQ parameters.
    pub eq: EqParams,
    /// Blended volume in dB.
    pub volume_db: f32,
    /// The highest-priority zone currently affecting the listener.
    pub dominant_zone: Option<ZoneId>,
}

impl BlendState {
    /// Create an initial "silent" blend state.
    pub fn silent() -> Self {
        Self {
            zone_weights: HashMap::new(),
            reverb: ReverbParams::outdoor(),
            eq: EqParams::flat(),
            volume_db: 0.0,
            dominant_zone: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotTransition
// ---------------------------------------------------------------------------

/// An active crossfade between two snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotTransition {
    /// Source snapshot.
    pub from: AudioSnapshot,
    /// Target snapshot.
    pub to: AudioSnapshot,
    /// Duration of the crossfade in seconds.
    pub duration: f32,
    /// Elapsed time.
    pub elapsed: f32,
    /// Whether this transition is complete.
    pub complete: bool,
}

impl SnapshotTransition {
    /// Create a new transition.
    pub fn new(from: AudioSnapshot, to: AudioSnapshot, duration: f32) -> Self {
        Self {
            from,
            to,
            duration: duration.max(0.001),
            elapsed: 0.0,
            complete: false,
        }
    }

    /// Advance the transition by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        self.elapsed += dt;
        if self.elapsed >= self.duration {
            self.elapsed = self.duration;
            self.complete = true;
        }
    }

    /// Returns the current interpolated snapshot.
    pub fn current(&self) -> AudioSnapshot {
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        AudioSnapshot::lerp(&self.from, &self.to, t)
    }

    /// Returns the progress [0, 1].
    pub fn progress(&self) -> f32 {
        (self.elapsed / self.duration).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// AudioZoneSystem
// ---------------------------------------------------------------------------

/// The main audio zone management system.
///
/// Manages zones, computes blend weights for the current listener position,
/// and handles snapshot transitions.
pub struct AudioZoneSystem {
    /// All registered zones.
    zones: Vec<AudioZone>,
    /// Next zone ID.
    next_zone_id: u32,
    /// Current blend state.
    blend_state: BlendState,
    /// Saved snapshots.
    snapshots: HashMap<String, AudioSnapshot>,
    /// Active snapshot transition.
    transition: Option<SnapshotTransition>,
    /// Default reverb for when no zone is active.
    default_reverb: ReverbParams,
    /// Default EQ for when no zone is active.
    default_eq: EqParams,
    /// Listener position.
    listener_pos: [f32; 3],
    /// Smoothing factor for blend weight changes [0, 1].
    /// Lower = slower transition.
    blend_smoothing: f32,
    /// Previous frame's zone weights (for smoothing).
    prev_weights: HashMap<ZoneId, f32>,
}

impl AudioZoneSystem {
    /// Create a new audio zone system.
    pub fn new() -> Self {
        Self {
            zones: Vec::new(),
            next_zone_id: 1,
            blend_state: BlendState::silent(),
            snapshots: HashMap::new(),
            transition: None,
            default_reverb: ReverbParams::outdoor(),
            default_eq: EqParams::flat(),
            listener_pos: [0.0, 0.0, 0.0],
            blend_smoothing: 0.1,
            prev_weights: HashMap::new(),
        }
    }

    /// Set the default reverb for when no zone is active.
    pub fn set_default_reverb(&mut self, reverb: ReverbParams) {
        self.default_reverb = reverb;
    }

    /// Set the default EQ.
    pub fn set_default_eq(&mut self, eq: EqParams) {
        self.default_eq = eq;
    }

    /// Set the blend smoothing factor.
    pub fn set_blend_smoothing(&mut self, factor: f32) {
        self.blend_smoothing = factor.clamp(0.01, 1.0);
    }

    // -------------------------------------------------------------------
    // Zone management
    // -------------------------------------------------------------------

    /// Add a zone and return its ID.
    pub fn add_zone(&mut self, zone: AudioZone) -> ZoneId {
        let id = zone.id;
        self.zones.push(zone);
        id
    }

    /// Create and add a zone, returning the auto-assigned ID.
    pub fn create_zone(
        &mut self,
        name: impl Into<String>,
        shape: ZoneShape,
        reverb: ReverbParams,
        eq: EqParams,
    ) -> ZoneId {
        let id = ZoneId(self.next_zone_id);
        self.next_zone_id += 1;

        let zone = AudioZone::new(id, name, shape, reverb, eq);
        self.zones.push(zone);
        id
    }

    /// Remove a zone by ID.
    pub fn remove_zone(&mut self, id: ZoneId) -> bool {
        let before = self.zones.len();
        self.zones.retain(|z| z.id != id);
        self.zones.len() < before
    }

    /// Get a zone by ID.
    pub fn zone(&self, id: ZoneId) -> Option<&AudioZone> {
        self.zones.iter().find(|z| z.id == id)
    }

    /// Get a mutable zone by ID.
    pub fn zone_mut(&mut self, id: ZoneId) -> Option<&mut AudioZone> {
        self.zones.iter_mut().find(|z| z.id == id)
    }

    /// Enable or disable a zone.
    pub fn set_zone_enabled(&mut self, id: ZoneId, enabled: bool) {
        if let Some(zone) = self.zone_mut(id) {
            zone.enabled = enabled;
        }
    }

    /// Returns the number of zones.
    pub fn zone_count(&self) -> usize {
        self.zones.len()
    }

    /// Returns all zone IDs.
    pub fn zone_ids(&self) -> Vec<ZoneId> {
        self.zones.iter().map(|z| z.id).collect()
    }

    // -------------------------------------------------------------------
    // Update
    // -------------------------------------------------------------------

    /// Update the listener position and recompute blend weights.
    pub fn update(&mut self, listener_pos: [f32; 3], dt: f32) {
        self.listener_pos = listener_pos;

        // Compute raw blend weights for all active zones.
        let mut raw_weights: Vec<(ZoneId, f32, i32)> = Vec::new();
        for zone in &self.zones {
            if !zone.enabled {
                continue;
            }
            let weight = zone.blend_weight(&listener_pos);
            if weight > 0.0 {
                raw_weights.push((zone.id, weight, zone.priority));
            }
        }

        // Sort by priority (descending).
        raw_weights.sort_by(|a, b| b.2.cmp(&a.2));

        // Normalize weights.
        let mut zone_weights: HashMap<ZoneId, f32> = HashMap::new();
        let total_weight: f32 = raw_weights.iter().map(|(_, w, _)| w).sum();

        if total_weight > 0.0 {
            for (id, weight, _) in &raw_weights {
                zone_weights.insert(*id, weight / total_weight);
            }
        }

        // Smooth weights over time.
        let smoothed = self.smooth_weights(&zone_weights, dt);
        self.prev_weights = smoothed.clone();

        // Compute blended reverb and EQ.
        let (blended_reverb, blended_eq, blended_volume) = self.blend_params(&smoothed);

        let dominant = raw_weights.first().map(|(id, _, _)| *id);

        self.blend_state = BlendState {
            zone_weights: smoothed,
            reverb: blended_reverb,
            eq: blended_eq,
            volume_db: blended_volume,
            dominant_zone: dominant,
        };

        // Update snapshot transition if active.
        if let Some(transition) = &mut self.transition {
            transition.update(dt);
            if transition.complete {
                // Apply the final snapshot.
                let final_snap = transition.current();
                self.blend_state.reverb = final_snap.reverb;
                self.blend_state.eq = final_snap.eq;
                self.blend_state.volume_db = final_snap.volume_db;
            }
        }

        if self.transition.as_ref().map_or(false, |t| t.complete) {
            self.transition = None;
        }
    }

    /// Smooth zone weights to avoid jarring changes.
    fn smooth_weights(
        &self,
        target: &HashMap<ZoneId, f32>,
        dt: f32,
    ) -> HashMap<ZoneId, f32> {
        let alpha = (self.blend_smoothing * dt * 60.0).clamp(0.0, 1.0);
        let mut result: HashMap<ZoneId, f32> = HashMap::new();

        // Blend towards target.
        for (&id, &target_w) in target {
            let prev_w = self.prev_weights.get(&id).copied().unwrap_or(0.0);
            let smoothed = prev_w + (target_w - prev_w) * alpha;
            result.insert(id, smoothed);
        }

        // Fade out zones no longer in target.
        for (&id, &prev_w) in &self.prev_weights {
            if !target.contains_key(&id) {
                let faded = prev_w * (1.0 - alpha);
                if faded > 0.001 {
                    result.insert(id, faded);
                }
            }
        }

        result
    }

    /// Compute blended reverb, EQ, and volume from zone weights.
    fn blend_params(
        &self,
        weights: &HashMap<ZoneId, f32>,
    ) -> (ReverbParams, EqParams, f32) {
        if weights.is_empty() {
            return (
                self.default_reverb.clone(),
                self.default_eq.clone(),
                0.0,
            );
        }

        let mut blended_reverb = self.default_reverb.clone();
        let mut blended_eq = self.default_eq.clone();
        let mut blended_volume = 0.0_f32;
        let mut total_w = 0.0_f32;

        for (&id, &weight) in weights {
            if let Some(zone) = self.zones.iter().find(|z| z.id == id) {
                if total_w == 0.0 {
                    blended_reverb = zone.reverb.clone();
                    blended_eq = zone.eq.clone();
                    blended_volume = zone.volume_db;
                } else {
                    let t = weight / (total_w + weight);
                    blended_reverb = ReverbParams::lerp(&blended_reverb, &zone.reverb, t);
                    blended_eq = EqParams::lerp(&blended_eq, &zone.eq, t);
                    blended_volume += (zone.volume_db - blended_volume) * t;
                }
                total_w += weight;
            }
        }

        (blended_reverb, blended_eq, blended_volume)
    }

    // -------------------------------------------------------------------
    // Snapshot system
    // -------------------------------------------------------------------

    /// Save the current audio state as a named snapshot.
    pub fn save_snapshot(&mut self, name: impl Into<String>) {
        let snapshot = AudioSnapshot::capture(
            name,
            &self.blend_state.reverb,
            &self.blend_state.eq,
            self.blend_state.volume_db,
            &self.blend_state.zone_weights,
        );
        self.snapshots.insert(snapshot.name.clone(), snapshot);
    }

    /// Restore a saved snapshot with a crossfade.
    pub fn restore_snapshot(&mut self, name: &str, duration: f32) -> bool {
        if let Some(target) = self.snapshots.get(name) {
            let current = AudioSnapshot::capture(
                "current",
                &self.blend_state.reverb,
                &self.blend_state.eq,
                self.blend_state.volume_db,
                &self.blend_state.zone_weights,
            );

            self.transition = Some(SnapshotTransition::new(
                current,
                target.clone(),
                duration,
            ));
            true
        } else {
            false
        }
    }

    /// Delete a saved snapshot.
    pub fn delete_snapshot(&mut self, name: &str) -> bool {
        self.snapshots.remove(name).is_some()
    }

    /// Returns the names of all saved snapshots.
    pub fn snapshot_names(&self) -> Vec<&str> {
        self.snapshots.keys().map(|k| k.as_str()).collect()
    }

    // -------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------

    /// Returns the current blend state.
    pub fn blend_state(&self) -> &BlendState {
        &self.blend_state
    }

    /// Returns the current blended reverb parameters.
    pub fn current_reverb(&self) -> &ReverbParams {
        &self.blend_state.reverb
    }

    /// Returns the current blended EQ parameters.
    pub fn current_eq(&self) -> &EqParams {
        &self.blend_state.eq
    }

    /// Returns the dominant zone ID.
    pub fn dominant_zone(&self) -> Option<ZoneId> {
        self.blend_state.dominant_zone
    }

    /// Returns whether a snapshot transition is in progress.
    pub fn is_transitioning(&self) -> bool {
        self.transition.is_some()
    }

    /// Returns the transition progress [0, 1] or None if not transitioning.
    pub fn transition_progress(&self) -> Option<f32> {
        self.transition.as_ref().map(|t| t.progress())
    }

    /// Returns the listener position.
    pub fn listener_pos(&self) -> [f32; 3] {
        self.listener_pos
    }
}

impl Default for AudioZoneSystem {
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
    fn test_zone_shape_box_contains() {
        let shape = ZoneShape::Box {
            center: [0.0, 0.0, 0.0],
            half_extents: [5.0, 5.0, 5.0],
        };
        assert!(shape.contains(&[0.0, 0.0, 0.0]));
        assert!(shape.contains(&[4.0, 4.0, 4.0]));
        assert!(!shape.contains(&[6.0, 0.0, 0.0]));
    }

    #[test]
    fn test_zone_shape_sphere_contains() {
        let shape = ZoneShape::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 5.0,
        };
        assert!(shape.contains(&[0.0, 0.0, 0.0]));
        assert!(shape.contains(&[3.0, 4.0, 0.0]));
        assert!(!shape.contains(&[4.0, 4.0, 4.0])); // dist ~6.93
    }

    #[test]
    fn test_zone_shape_blend_weight() {
        let shape = ZoneShape::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 10.0,
        };

        // Center — fully inside.
        let w = shape.blend_weight(&[0.0, 0.0, 0.0], 2.0);
        assert!((w - 1.0).abs() < 0.01);

        // Outside.
        let w = shape.blend_weight(&[20.0, 0.0, 0.0], 2.0);
        assert!((w - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_reverb_presets() {
        let small = ReverbParams::small_room();
        assert!(small.room_size < 0.5);
        assert!(small.decay_time < 1.0);

        let hall = ReverbParams::large_hall();
        assert!(hall.room_size > 0.5);
        assert!(hall.decay_time > 1.0);
    }

    #[test]
    fn test_reverb_lerp() {
        let a = ReverbParams::small_room();
        let b = ReverbParams::large_hall();
        let mid = ReverbParams::lerp(&a, &b, 0.5);

        assert!(mid.room_size > a.room_size);
        assert!(mid.room_size < b.room_size);
    }

    #[test]
    fn test_eq_presets() {
        let indoor = EqParams::indoor();
        assert!(indoor.low_gain_db > 0.0);

        let outdoor = EqParams::outdoor();
        assert!(outdoor.high_gain_db > 0.0);
    }

    #[test]
    fn test_zone_system_basic() {
        let mut system = AudioZoneSystem::new();

        let id = system.create_zone(
            "office",
            ZoneShape::Box {
                center: [0.0, 0.0, 0.0],
                half_extents: [10.0, 5.0, 10.0],
            },
            ReverbParams::small_room(),
            EqParams::indoor(),
        );

        assert_eq!(system.zone_count(), 1);
        assert!(system.zone(id).is_some());
    }

    #[test]
    fn test_zone_system_update() {
        let mut system = AudioZoneSystem::new();

        system.create_zone(
            "room",
            ZoneShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 10.0,
            },
            ReverbParams::small_room(),
            EqParams::indoor(),
        );

        // Listener inside the zone.
        system.update([0.0, 0.0, 0.0], 1.0 / 60.0);
        assert!(system.dominant_zone().is_some());
    }

    #[test]
    fn test_zone_system_outside() {
        let mut system = AudioZoneSystem::new();

        system.create_zone(
            "room",
            ZoneShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 5.0,
            },
            ReverbParams::small_room(),
            EqParams::indoor(),
        );

        system.update([100.0, 100.0, 100.0], 1.0 / 60.0);
        assert!(system.dominant_zone().is_none());
    }

    #[test]
    fn test_snapshot_save_restore() {
        let mut system = AudioZoneSystem::new();

        system.create_zone(
            "room",
            ZoneShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 10.0,
            },
            ReverbParams::small_room(),
            EqParams::indoor(),
        );

        system.update([0.0, 0.0, 0.0], 1.0 / 60.0);
        system.save_snapshot("state_1");

        assert!(system.snapshot_names().contains(&"state_1"));
        assert!(system.restore_snapshot("state_1", 1.0));
        assert!(system.is_transitioning());
    }

    #[test]
    fn test_snapshot_transition() {
        let from = AudioSnapshot {
            name: "from".to_string(),
            reverb: ReverbParams::small_room(),
            eq: EqParams::indoor(),
            volume_db: 0.0,
            zone_weights: HashMap::new(),
        };

        let to = AudioSnapshot {
            name: "to".to_string(),
            reverb: ReverbParams::large_hall(),
            eq: EqParams::outdoor(),
            volume_db: -3.0,
            zone_weights: HashMap::new(),
        };

        let mut transition = SnapshotTransition::new(from, to, 1.0);
        assert!(!transition.complete);

        transition.update(0.5);
        assert!((transition.progress() - 0.5).abs() < 0.01);

        transition.update(0.6);
        assert!(transition.complete);
    }

    #[test]
    fn test_zone_enable_disable() {
        let mut system = AudioZoneSystem::new();
        let id = system.create_zone(
            "test",
            ZoneShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 10.0,
            },
            ReverbParams::new(),
            EqParams::flat(),
        );

        system.set_zone_enabled(id, false);
        assert!(!system.zone(id).unwrap().enabled);
    }

    #[test]
    fn test_signed_distance_sphere() {
        let shape = ZoneShape::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 5.0,
        };

        // Inside.
        let sd = shape.signed_distance(&[0.0, 0.0, 0.0]);
        assert!(sd < 0.0);

        // On surface.
        let sd = shape.signed_distance(&[5.0, 0.0, 0.0]);
        assert!(sd.abs() < 0.01);

        // Outside.
        let sd = shape.signed_distance(&[10.0, 0.0, 0.0]);
        assert!(sd > 0.0);
    }
}
