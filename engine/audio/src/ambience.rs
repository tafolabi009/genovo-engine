//! Ambient sound system with zone-based activation, crossfading,
//! layered ambient loops, and randomised one-shot sounds.
//!
//! Provides:
//! - `AmbienceZone`: 3D volume (AABB or sphere) with ambient sound layers
//! - `RandomOneShot`: periodic random sound playback with timing/pitch variation
//! - `AmbienceManager`: tracks listener position, activates/deactivates zones
//! - `AmbienceComponent`, `AmbienceSystem` for ECS integration

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default crossfade duration when entering/leaving a zone.
const DEFAULT_ZONE_CROSSFADE: f32 = 2.0;
/// Default activation distance margin beyond the zone boundary.
const DEFAULT_ACTIVATION_MARGIN: f32 = 5.0;

// ---------------------------------------------------------------------------
// Zone shape
// ---------------------------------------------------------------------------

/// Shape of an ambient sound zone.
#[derive(Debug, Clone)]
pub enum ZoneShape {
    /// Axis-aligned bounding box.
    AABB { min: Vec3, max: Vec3 },
    /// Sphere.
    Sphere { center: Vec3, radius: f32 },
}

impl ZoneShape {
    /// Check if a point is inside the zone.
    pub fn contains(&self, point: Vec3) -> bool {
        match self {
            ZoneShape::AABB { min, max } => {
                point.x >= min.x
                    && point.x <= max.x
                    && point.y >= min.y
                    && point.y <= max.y
                    && point.z >= min.z
                    && point.z <= max.z
            }
            ZoneShape::Sphere { center, radius } => {
                (point - *center).length_squared() <= radius * radius
            }
        }
    }

    /// Compute the signed distance from a point to the zone boundary.
    /// Negative = inside, positive = outside.
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        match self {
            ZoneShape::AABB { min, max } => {
                let center = (*min + *max) * 0.5;
                let half = (*max - *min) * 0.5;
                let d = (point - center).abs() - half;
                let outside = Vec3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0)).length();
                let inside = d.x.max(d.y).max(d.z).min(0.0);
                outside + inside
            }
            ZoneShape::Sphere { center, radius } => {
                (point - *center).length() - radius
            }
        }
    }

    /// Compute a blend factor [0, 1] based on distance from the boundary.
    /// 1.0 = fully inside, 0.0 = outside the crossfade margin.
    pub fn blend_factor(&self, point: Vec3, crossfade_distance: f32) -> f32 {
        let dist = self.signed_distance(point);
        if dist <= 0.0 {
            1.0 // Fully inside
        } else if dist >= crossfade_distance {
            0.0 // Fully outside
        } else {
            1.0 - dist / crossfade_distance
        }
    }
}

// ---------------------------------------------------------------------------
// Ambient layer
// ---------------------------------------------------------------------------

/// A continuous sound layer within an ambient zone.
#[derive(Debug, Clone)]
pub struct AmbientLayer {
    /// Name of this layer (e.g. "wind", "birds", "water").
    pub name: String,
    /// Clip name to loop.
    pub clip_name: String,
    /// Base volume [0, 1].
    pub volume: f32,
    /// Current effective volume (after zone blending).
    pub effective_volume: f32,
    /// Pitch multiplier.
    pub pitch: f32,
    /// Whether this layer is playing.
    pub playing: bool,
}

impl AmbientLayer {
    /// Create a new ambient layer.
    pub fn new(name: &str, clip_name: &str, volume: f32) -> Self {
        Self {
            name: name.to_string(),
            clip_name: clip_name.to_string(),
            volume,
            effective_volume: 0.0,
            pitch: 1.0,
            playing: false,
        }
    }

    /// Set volume.
    pub fn with_volume(mut self, volume: f32) -> Self {
        self.volume = volume;
        self
    }

    /// Set pitch.
    pub fn with_pitch(mut self, pitch: f32) -> Self {
        self.pitch = pitch;
        self
    }
}

// ---------------------------------------------------------------------------
// Random one-shot
// ---------------------------------------------------------------------------

/// A random one-shot sound that plays periodically with variation.
///
/// Used for intermittent ambient sounds like bird chirps, twig snaps,
/// distant thunder, etc.
#[derive(Debug, Clone)]
pub struct RandomOneShot {
    /// Available clip names (one is chosen at random each trigger).
    pub clips: Vec<String>,
    /// Minimum interval between triggers (seconds).
    pub min_interval: f32,
    /// Maximum interval between triggers (seconds).
    pub max_interval: f32,
    /// Minimum volume.
    pub volume_min: f32,
    /// Maximum volume.
    pub volume_max: f32,
    /// Minimum pitch multiplier.
    pub pitch_min: f32,
    /// Maximum pitch multiplier.
    pub pitch_max: f32,
    /// Spatial radius: if > 0, the sound is positioned randomly within this
    /// radius of the zone center.
    pub spatial_radius: f32,
    /// Time until the next trigger.
    next_trigger_time: f32,
    /// Elapsed time since the last trigger.
    elapsed: f32,
    /// Simple counter for pseudo-random variation.
    rng_counter: u32,
    /// Whether this one-shot generator is enabled.
    pub enabled: bool,
}

impl RandomOneShot {
    /// Create a new random one-shot generator.
    pub fn new(clips: Vec<String>, min_interval: f32, max_interval: f32) -> Self {
        Self {
            clips,
            min_interval,
            max_interval,
            volume_min: 0.8,
            volume_max: 1.0,
            pitch_min: 0.9,
            pitch_max: 1.1,
            spatial_radius: 0.0,
            next_trigger_time: min_interval,
            elapsed: 0.0,
            rng_counter: 0,
            enabled: true,
        }
    }

    /// Set volume range.
    pub fn with_volume_range(mut self, min: f32, max: f32) -> Self {
        self.volume_min = min;
        self.volume_max = max;
        self
    }

    /// Set pitch range.
    pub fn with_pitch_range(mut self, min: f32, max: f32) -> Self {
        self.pitch_min = min;
        self.pitch_max = max;
        self
    }

    /// Set spatial radius.
    pub fn with_spatial_radius(mut self, radius: f32) -> Self {
        self.spatial_radius = radius;
        self
    }

    /// Update the one-shot timer. Returns `Some(trigger_info)` when it fires.
    pub fn update(&mut self, dt: f32) -> Option<OneShotTrigger> {
        if !self.enabled || self.clips.is_empty() {
            return None;
        }

        self.elapsed += dt;

        if self.elapsed >= self.next_trigger_time {
            self.elapsed = 0.0;
            self.rng_counter = self.rng_counter.wrapping_add(1);

            // Pick next interval
            let t = self.pseudo_random();
            self.next_trigger_time =
                self.min_interval + t * (self.max_interval - self.min_interval);

            // Pick clip
            let clip_idx = (self.rng_counter as usize) % self.clips.len();
            let clip_name = self.clips[clip_idx].clone();

            // Pick volume and pitch
            let t2 = self.pseudo_random();
            let volume = self.volume_min + t2 * (self.volume_max - self.volume_min);
            let t3 = self.pseudo_random();
            let pitch = self.pitch_min + t3 * (self.pitch_max - self.pitch_min);

            // Pick spatial offset
            let offset = if self.spatial_radius > 0.0 {
                let t4 = self.pseudo_random();
                let t5 = self.pseudo_random();
                let angle = t4 * std::f32::consts::TAU;
                let r = t5 * self.spatial_radius;
                Vec3::new(angle.cos() * r, 0.0, angle.sin() * r)
            } else {
                Vec3::ZERO
            };

            return Some(OneShotTrigger {
                clip_name,
                volume,
                pitch,
                spatial_offset: offset,
            });
        }

        None
    }

    fn pseudo_random(&mut self) -> f32 {
        self.rng_counter = self.rng_counter.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng_counter >> 16) as f32 / 65536.0
    }
}

/// Information about a triggered one-shot sound.
#[derive(Debug, Clone)]
pub struct OneShotTrigger {
    /// Clip name to play.
    pub clip_name: String,
    /// Volume for this instance.
    pub volume: f32,
    /// Pitch for this instance.
    pub pitch: f32,
    /// Spatial offset from the zone center.
    pub spatial_offset: Vec3,
}

// ---------------------------------------------------------------------------
// AmbienceZone
// ---------------------------------------------------------------------------

/// A 3D volume defining an ambient sound region.
///
/// When the listener enters the zone, ambient layers begin playing with
/// a crossfade. Random one-shots are triggered periodically.
#[derive(Debug, Clone)]
pub struct AmbienceZone {
    /// Unique name for this zone.
    pub name: String,
    /// The zone's 3D shape.
    pub shape: ZoneShape,
    /// Continuous ambient layers.
    pub layers: Vec<AmbientLayer>,
    /// Random one-shot generators.
    pub one_shots: Vec<RandomOneShot>,
    /// Crossfade distance (how far beyond the boundary to fade).
    pub crossfade_distance: f32,
    /// Priority (higher priority zones take precedence when overlapping).
    pub priority: i32,
    /// Whether this zone is active.
    pub active: bool,
    /// Current blend factor [0, 1] based on listener position.
    pub blend_factor: f32,
    /// Master volume multiplier for this zone.
    pub volume: f32,
}

impl AmbienceZone {
    /// Create a new AABB-shaped ambience zone.
    pub fn new_aabb(name: &str, min: Vec3, max: Vec3) -> Self {
        Self {
            name: name.to_string(),
            shape: ZoneShape::AABB { min, max },
            layers: Vec::new(),
            one_shots: Vec::new(),
            crossfade_distance: DEFAULT_ZONE_CROSSFADE,
            priority: 0,
            active: true,
            blend_factor: 0.0,
            volume: 1.0,
        }
    }

    /// Create a new sphere-shaped ambience zone.
    pub fn new_sphere(name: &str, center: Vec3, radius: f32) -> Self {
        Self {
            name: name.to_string(),
            shape: ZoneShape::Sphere { center, radius },
            layers: Vec::new(),
            one_shots: Vec::new(),
            crossfade_distance: DEFAULT_ZONE_CROSSFADE,
            priority: 0,
            active: true,
            blend_factor: 0.0,
            volume: 1.0,
        }
    }

    /// Add an ambient layer.
    pub fn add_layer(&mut self, layer: AmbientLayer) {
        self.layers.push(layer);
    }

    /// Add a random one-shot generator.
    pub fn add_one_shot(&mut self, one_shot: RandomOneShot) {
        self.one_shots.push(one_shot);
    }

    /// Update the zone based on listener position.
    ///
    /// Returns any triggered one-shot sounds.
    pub fn update(&mut self, listener_pos: Vec3, dt: f32) -> Vec<OneShotTrigger> {
        if !self.active {
            self.blend_factor = 0.0;
            return Vec::new();
        }

        // Compute blend factor
        self.blend_factor = self.shape.blend_factor(listener_pos, self.crossfade_distance);

        // Update layer effective volumes
        for layer in &mut self.layers {
            layer.effective_volume = layer.volume * self.blend_factor * self.volume;
            layer.playing = layer.effective_volume > 0.001;
        }

        // Update one-shots (only if blend > 0)
        let mut triggers = Vec::new();
        if self.blend_factor > 0.01 {
            for one_shot in &mut self.one_shots {
                if let Some(mut trigger) = one_shot.update(dt) {
                    trigger.volume *= self.blend_factor * self.volume;
                    triggers.push(trigger);
                }
            }
        }

        triggers
    }

    /// Whether the listener is within activation range.
    pub fn is_listener_nearby(&self, listener_pos: Vec3) -> bool {
        self.shape.signed_distance(listener_pos)
            < self.crossfade_distance + DEFAULT_ACTIVATION_MARGIN
    }
}

// ---------------------------------------------------------------------------
// AmbienceManager
// ---------------------------------------------------------------------------

/// Manages multiple ambience zones, tracking the listener position and
/// activating/deactivating zones with smooth transitions.
#[derive(Debug, Clone)]
pub struct AmbienceManager {
    /// All registered ambience zones.
    pub zones: Vec<AmbienceZone>,
    /// Current listener position.
    pub listener_position: Vec3,
    /// Triggered one-shots from the most recent update.
    pub pending_triggers: Vec<OneShotTrigger>,
    /// Master volume for all ambient sounds.
    pub master_volume: f32,
}

impl AmbienceManager {
    /// Create a new ambience manager.
    pub fn new() -> Self {
        Self {
            zones: Vec::new(),
            listener_position: Vec3::ZERO,
            pending_triggers: Vec::new(),
            master_volume: 1.0,
        }
    }

    /// Add a zone.
    pub fn add_zone(&mut self, zone: AmbienceZone) {
        self.zones.push(zone);
    }

    /// Remove a zone by name.
    pub fn remove_zone(&mut self, name: &str) {
        self.zones.retain(|z| z.name != name);
    }

    /// Get a zone by name.
    pub fn get_zone(&self, name: &str) -> Option<&AmbienceZone> {
        self.zones.iter().find(|z| z.name == name)
    }

    /// Get a mutable reference to a zone by name.
    pub fn get_zone_mut(&mut self, name: &str) -> Option<&mut AmbienceZone> {
        self.zones.iter_mut().find(|z| z.name == name)
    }

    /// Update the listener position and process all zones.
    pub fn update(&mut self, listener_pos: Vec3, dt: f32) {
        self.listener_position = listener_pos;
        self.pending_triggers.clear();

        for zone in &mut self.zones {
            let triggers = zone.update(listener_pos, dt);
            for mut t in triggers {
                t.volume *= self.master_volume;
                self.pending_triggers.push(t);
            }
        }
    }

    /// Get all zones that are currently active (blend > 0).
    pub fn active_zones(&self) -> Vec<&AmbienceZone> {
        self.zones
            .iter()
            .filter(|z| z.blend_factor > 0.001)
            .collect()
    }

    /// Drain pending one-shot triggers.
    pub fn drain_triggers(&mut self) -> Vec<OneShotTrigger> {
        std::mem::take(&mut self.pending_triggers)
    }
}

impl Default for AmbienceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for ambient sound management.
pub struct AmbienceComponent {
    /// The ambience manager.
    pub manager: AmbienceManager,
    /// Whether this component is active.
    pub active: bool,
}

impl AmbienceComponent {
    /// Create a new ambience component.
    pub fn new() -> Self {
        Self {
            manager: AmbienceManager::new(),
            active: true,
        }
    }
}

impl Default for AmbienceComponent {
    fn default() -> Self {
        Self::new()
    }
}

/// System that updates ambient sound components each frame.
pub struct AmbienceSystem {
    _placeholder: (),
}

impl Default for AmbienceSystem {
    fn default() -> Self {
        Self { _placeholder: () }
    }
}

impl AmbienceSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update all ambience components.
    pub fn update(&self, dt: f32, listener_pos: Vec3, components: &mut [AmbienceComponent]) {
        for comp in components.iter_mut() {
            if comp.active {
                comp.manager.update(listener_pos, dt);
            }
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_zone_contains() {
        let shape = ZoneShape::AABB {
            min: Vec3::new(-5.0, -5.0, -5.0),
            max: Vec3::new(5.0, 5.0, 5.0),
        };

        assert!(shape.contains(Vec3::ZERO));
        assert!(shape.contains(Vec3::new(4.0, 4.0, 4.0)));
        assert!(!shape.contains(Vec3::new(10.0, 0.0, 0.0)));
    }

    #[test]
    fn test_sphere_zone_contains() {
        let shape = ZoneShape::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        };

        assert!(shape.contains(Vec3::ZERO));
        assert!(shape.contains(Vec3::new(3.0, 0.0, 0.0)));
        assert!(!shape.contains(Vec3::new(10.0, 0.0, 0.0)));
    }

    #[test]
    fn test_zone_blend_factor() {
        let shape = ZoneShape::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        };

        // Inside: blend = 1.0
        assert!((shape.blend_factor(Vec3::ZERO, 2.0) - 1.0).abs() < 0.01);

        // On boundary: blend = 1.0 (signed_distance = 0)
        let on_boundary = shape.blend_factor(Vec3::new(5.0, 0.0, 0.0), 2.0);
        assert!((on_boundary - 1.0).abs() < 0.01);

        // Outside by crossfade distance: blend = 0.0
        let outside = shape.blend_factor(Vec3::new(7.0, 0.0, 0.0), 2.0);
        assert!(outside.abs() < 0.01);

        // Midway in crossfade: blend ~ 0.5
        let mid = shape.blend_factor(Vec3::new(6.0, 0.0, 0.0), 2.0);
        assert!((mid - 0.5).abs() < 0.1, "Mid blend = {}", mid);
    }

    #[test]
    fn test_ambience_zone_update() {
        let mut zone = AmbienceZone::new_sphere("forest", Vec3::ZERO, 10.0);
        zone.add_layer(AmbientLayer::new("wind", "wind_loop", 0.7));
        zone.add_layer(AmbientLayer::new("birds", "birds_loop", 0.5));

        // Listener inside zone
        let triggers = zone.update(Vec3::new(1.0, 0.0, 0.0), 1.0 / 60.0);

        assert!((zone.blend_factor - 1.0).abs() < 0.01);
        assert!(zone.layers[0].effective_volume > 0.0);
        assert!(zone.layers[0].playing);
    }

    #[test]
    fn test_ambience_zone_outside() {
        let mut zone = AmbienceZone::new_sphere("forest", Vec3::ZERO, 5.0);
        zone.add_layer(AmbientLayer::new("wind", "wind_loop", 0.7));

        // Listener far outside zone
        zone.update(Vec3::new(100.0, 0.0, 0.0), 1.0 / 60.0);

        assert!(zone.blend_factor < 0.01);
        assert!(!zone.layers[0].playing);
    }

    #[test]
    fn test_random_one_shot() {
        let mut one_shot = RandomOneShot::new(
            vec!["chirp_1".into(), "chirp_2".into(), "chirp_3".into()],
            0.1,
            0.2,
        );

        let mut trigger_count = 0;
        for _ in 0..300 {
            if one_shot.update(1.0 / 60.0).is_some() {
                trigger_count += 1;
            }
        }

        // Over 5 seconds with 0.1-0.2s interval, expect ~25-50 triggers
        assert!(
            trigger_count > 10 && trigger_count < 100,
            "Got {} triggers",
            trigger_count
        );
    }

    #[test]
    fn test_ambience_manager() {
        let mut manager = AmbienceManager::new();

        let mut zone = AmbienceZone::new_sphere("forest", Vec3::ZERO, 10.0);
        zone.add_layer(AmbientLayer::new("wind", "wind_loop", 0.7));
        manager.add_zone(zone);

        manager.update(Vec3::new(1.0, 0.0, 0.0), 1.0 / 60.0);

        let active = manager.active_zones();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "forest");
    }

    #[test]
    fn test_ambience_manager_no_active() {
        let mut manager = AmbienceManager::new();
        let zone = AmbienceZone::new_sphere("forest", Vec3::ZERO, 5.0);
        manager.add_zone(zone);

        manager.update(Vec3::new(100.0, 0.0, 0.0), 1.0 / 60.0);

        assert!(manager.active_zones().is_empty());
    }

    #[test]
    fn test_ambience_component() {
        let comp = AmbienceComponent::new();
        assert!(comp.active);
    }

    #[test]
    fn test_ambience_system() {
        let system = AmbienceSystem::new();
        let mut comps = vec![AmbienceComponent::new()];
        comps[0].manager.add_zone(AmbienceZone::new_sphere(
            "test",
            Vec3::ZERO,
            5.0,
        ));

        system.update(1.0 / 60.0, Vec3::ZERO, &mut comps);
        // Should not panic
    }

    #[test]
    fn test_signed_distance_aabb() {
        let shape = ZoneShape::AABB {
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(1.0, 1.0, 1.0),
        };

        let inside = shape.signed_distance(Vec3::ZERO);
        assert!(inside < 0.0, "Inside should be negative: {}", inside);

        let outside = shape.signed_distance(Vec3::new(3.0, 0.0, 0.0));
        assert!(outside > 0.0, "Outside should be positive: {}", outside);
    }

    #[test]
    fn test_one_shot_volume_pitch_range() {
        let mut one_shot = RandomOneShot::new(vec!["clip".into()], 0.01, 0.02)
            .with_volume_range(0.3, 0.7)
            .with_pitch_range(0.8, 1.2);

        let mut volumes = Vec::new();
        let mut pitches = Vec::new();

        for _ in 0..1000 {
            if let Some(trigger) = one_shot.update(1.0 / 60.0) {
                volumes.push(trigger.volume);
                pitches.push(trigger.pitch);
            }
        }

        assert!(!volumes.is_empty());
        for &v in &volumes {
            assert!(v >= 0.29 && v <= 0.71, "Volume out of range: {}", v);
        }
        for &p in &pitches {
            assert!(p >= 0.79 && p <= 1.21, "Pitch out of range: {}", p);
        }
    }
}
