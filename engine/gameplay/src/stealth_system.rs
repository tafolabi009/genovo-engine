//! Stealth mechanics system.
//!
//! Provides visibility metering, light-level detection, noise generation
//! from movement and actions, last-known-position tracking, search patterns
//! for alerted AI, a disguise system, and line-of-sight checks that
//! integrate with the AI perception module.
//!
//! # Key concepts
//!
//! - **StealthComponent**: Per-entity component tracking visibility, noise
//!   output, and alert state.
//! - **LightProbe**: Samples the environment light level at a position to
//!   determine how visible an entity is.
//! - **NoiseEvent**: A sound emitted by movement, actions, or combat that
//!   can alert nearby AI.
//! - **SearchPattern**: Behavior that AI follows after losing sight of a
//!   target (search last known position, expand search area, etc.).
//! - **Disguise**: A disguise that reduces detection by specific factions.
//! - **StealthSystem**: Top-level manager integrating all stealth mechanics.

use std::collections::{HashMap, VecDeque};

use glam::Vec3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum visibility (fully visible).
pub const MAX_VISIBILITY: f32 = 100.0;

/// Minimum visibility (fully hidden).
pub const MIN_VISIBILITY: f32 = 0.0;

/// Default detection threshold (visibility above this triggers detection).
pub const DEFAULT_DETECTION_THRESHOLD: f32 = 50.0;

/// Light level considered full darkness (0..1).
pub const FULL_DARKNESS: f32 = 0.0;

/// Light level considered full daylight (0..1).
pub const FULL_LIGHT: f32 = 1.0;

/// Noise decay rate per second.
pub const NOISE_DECAY_RATE: f32 = 20.0;

/// Maximum noise level.
pub const MAX_NOISE: f32 = 100.0;

/// Noise threshold for AI awareness.
pub const NOISE_AWARENESS_THRESHOLD: f32 = 20.0;

/// Noise threshold for full alert.
pub const NOISE_ALERT_THRESHOLD: f32 = 60.0;

/// Maximum number of last-known positions tracked per entity.
pub const MAX_LKP_HISTORY: usize = 16;

/// Search pattern default duration (seconds).
pub const DEFAULT_SEARCH_DURATION: f32 = 30.0;

/// Maximum search radius expansion (world units).
pub const MAX_SEARCH_RADIUS: f32 = 50.0;

/// Search radius expansion rate per second.
pub const SEARCH_EXPANSION_RATE: f32 = 2.0;

/// Maximum disguises per entity.
pub const MAX_DISGUISES: usize = 4;

/// Default line-of-sight check interval (seconds).
pub const DEFAULT_LOS_CHECK_INTERVAL: f32 = 0.2;

/// Crouch visibility reduction (fraction).
pub const CROUCH_VISIBILITY_REDUCTION: f32 = 0.3;

/// Prone visibility reduction (fraction).
pub const PRONE_VISIBILITY_REDUCTION: f32 = 0.6;

/// Movement speed visibility scaling factor.
pub const MOVEMENT_VISIBILITY_FACTOR: f32 = 0.5;

/// Maximum stealth entities tracked.
pub const MAX_STEALTH_ENTITIES: usize = 512;

/// Maximum noise events per tick.
pub const MAX_NOISE_EVENTS_PER_TICK: usize = 256;

// ---------------------------------------------------------------------------
// AlertState
// ---------------------------------------------------------------------------

/// The AI's alert state regarding a stealthy target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertState {
    /// Unaware of any threat.
    Unaware,
    /// Suspicious — heard a noise or saw something briefly.
    Suspicious,
    /// Searching — actively looking for a detected target.
    Searching,
    /// Alert — knows the target's approximate position.
    Alert,
    /// Engaged — in direct combat.
    Engaged,
}

impl AlertState {
    /// Whether the entity is in a heightened state.
    pub fn is_alerted(&self) -> bool {
        !matches!(self, Self::Unaware)
    }

    /// Detection sensitivity multiplier (higher = easier to detect).
    pub fn sensitivity_multiplier(&self) -> f32 {
        match self {
            Self::Unaware => 1.0,
            Self::Suspicious => 1.5,
            Self::Searching => 2.0,
            Self::Alert => 2.5,
            Self::Engaged => 3.0,
        }
    }

    /// Movement speed multiplier for the alerted AI.
    pub fn movement_speed_multiplier(&self) -> f32 {
        match self {
            Self::Unaware => 1.0,
            Self::Suspicious => 1.0,
            Self::Searching => 1.2,
            Self::Alert => 1.5,
            Self::Engaged => 1.8,
        }
    }

    /// Label for UI.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Unaware => "Unaware",
            Self::Suspicious => "Suspicious",
            Self::Searching => "Searching",
            Self::Alert => "Alert",
            Self::Engaged => "Engaged",
        }
    }

    /// Color for UI indicator.
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            Self::Unaware => (100, 100, 100),
            Self::Suspicious => (255, 255, 0),
            Self::Searching => (255, 165, 0),
            Self::Alert => (255, 80, 0),
            Self::Engaged => (255, 0, 0),
        }
    }
}

// ---------------------------------------------------------------------------
// Posture
// ---------------------------------------------------------------------------

/// Body posture affecting visibility and noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Posture {
    Standing,
    Crouching,
    Prone,
}

impl Posture {
    /// Visibility reduction factor (0 = no reduction).
    pub fn visibility_reduction(&self) -> f32 {
        match self {
            Self::Standing => 0.0,
            Self::Crouching => CROUCH_VISIBILITY_REDUCTION,
            Self::Prone => PRONE_VISIBILITY_REDUCTION,
        }
    }

    /// Noise reduction factor.
    pub fn noise_reduction(&self) -> f32 {
        match self {
            Self::Standing => 0.0,
            Self::Crouching => 0.4,
            Self::Prone => 0.7,
        }
    }

    /// Movement speed modifier.
    pub fn speed_modifier(&self) -> f32 {
        match self {
            Self::Standing => 1.0,
            Self::Crouching => 0.5,
            Self::Prone => 0.2,
        }
    }

    /// Height offset for line-of-sight (meters).
    pub fn eye_height(&self) -> f32 {
        match self {
            Self::Standing => 1.7,
            Self::Crouching => 1.0,
            Self::Prone => 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// SurfaceType
// ---------------------------------------------------------------------------

/// Surface material affecting footstep noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceType {
    Grass,
    Dirt,
    Stone,
    Wood,
    Metal,
    Water,
    Gravel,
    Sand,
    Snow,
    Carpet,
}

impl SurfaceType {
    /// Noise multiplier for footsteps on this surface.
    pub fn noise_multiplier(&self) -> f32 {
        match self {
            Self::Grass => 0.5,
            Self::Dirt => 0.6,
            Self::Stone => 0.8,
            Self::Wood => 0.9,
            Self::Metal => 1.2,
            Self::Water => 1.0,
            Self::Gravel => 1.1,
            Self::Sand => 0.4,
            Self::Snow => 0.7,
            Self::Carpet => 0.3,
        }
    }

    /// Whether this surface leaves tracks.
    pub fn leaves_tracks(&self) -> bool {
        matches!(
            self,
            Self::Snow | Self::Sand | Self::Dirt | Self::Gravel
        )
    }
}

// ---------------------------------------------------------------------------
// NoiseEvent
// ---------------------------------------------------------------------------

/// A sound event that can alert AI.
#[derive(Debug, Clone)]
pub struct NoiseEvent {
    /// Source entity ID.
    pub source: u64,
    /// World position of the noise.
    pub position: Vec3,
    /// Volume/intensity (0..100).
    pub volume: f32,
    /// Type of noise.
    pub noise_type: NoiseType,
    /// Radius at which this noise can be heard.
    pub audible_radius: f32,
    /// Timestamp.
    pub timestamp: f64,
}

/// Type of noise event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoiseType {
    /// Footsteps.
    Footstep,
    /// Landing from a jump or fall.
    Landing,
    /// Weapon fire.
    Gunshot,
    /// Melee weapon swing.
    MeleeSwing,
    /// Object breaking (glass, crate, etc.).
    Breaking,
    /// Door opening/closing.
    Door,
    /// Voice/shout.
    Voice,
    /// Equipment jingle (armor, keys).
    Equipment,
    /// Environmental (falling debris, etc.).
    Environmental,
    /// Explosion.
    Explosion,
    /// Custom noise source.
    Custom,
}

impl NoiseType {
    /// Base volume for this noise type.
    pub fn base_volume(&self) -> f32 {
        match self {
            Self::Footstep => 15.0,
            Self::Landing => 30.0,
            Self::Gunshot => 90.0,
            Self::MeleeSwing => 25.0,
            Self::Breaking => 50.0,
            Self::Door => 20.0,
            Self::Voice => 35.0,
            Self::Equipment => 10.0,
            Self::Environmental => 40.0,
            Self::Explosion => 100.0,
            Self::Custom => 20.0,
        }
    }

    /// Base audible radius (world units).
    pub fn base_radius(&self) -> f32 {
        match self {
            Self::Footstep => 10.0,
            Self::Landing => 15.0,
            Self::Gunshot => 80.0,
            Self::MeleeSwing => 8.0,
            Self::Breaking => 25.0,
            Self::Door => 12.0,
            Self::Voice => 20.0,
            Self::Equipment => 5.0,
            Self::Environmental => 30.0,
            Self::Explosion => 100.0,
            Self::Custom => 15.0,
        }
    }
}

impl NoiseEvent {
    /// Create a noise event with default values for the type.
    pub fn new(source: u64, position: Vec3, noise_type: NoiseType, timestamp: f64) -> Self {
        Self {
            source,
            position,
            volume: noise_type.base_volume(),
            noise_type,
            audible_radius: noise_type.base_radius(),
            timestamp,
        }
    }

    /// Create with custom volume.
    pub fn with_volume(mut self, volume: f32) -> Self {
        self.volume = volume.clamp(0.0, MAX_NOISE);
        self
    }

    /// Check if a listener at a given position can hear this noise.
    pub fn audible_at(&self, listener_pos: Vec3) -> bool {
        listener_pos.distance(self.position) <= self.audible_radius
    }

    /// Get the volume as perceived at a given position.
    pub fn volume_at(&self, listener_pos: Vec3) -> f32 {
        let dist = listener_pos.distance(self.position);
        if dist >= self.audible_radius {
            return 0.0;
        }
        let falloff = 1.0 - (dist / self.audible_radius);
        self.volume * falloff * falloff
    }
}

// ---------------------------------------------------------------------------
// LastKnownPositionEntry
// ---------------------------------------------------------------------------

/// A record of where a target was last seen.
#[derive(Debug, Clone)]
pub struct LastKnownPositionEntry {
    /// Target entity ID.
    pub target_id: u64,
    /// Last known position.
    pub position: Vec3,
    /// Last known velocity/direction.
    pub velocity: Vec3,
    /// Timestamp when last seen.
    pub timestamp: f64,
    /// Confidence in this position (0..1, decays over time).
    pub confidence: f32,
    /// Whether a search has been conducted at this position.
    pub searched: bool,
}

impl LastKnownPositionEntry {
    /// Create a new LKP entry.
    pub fn new(target_id: u64, position: Vec3, velocity: Vec3, timestamp: f64) -> Self {
        Self {
            target_id,
            position,
            velocity,
            timestamp,
            confidence: 1.0,
            searched: false,
        }
    }

    /// Predict where the target might be now.
    pub fn predicted_position(&self, current_time: f64) -> Vec3 {
        let elapsed = (current_time - self.timestamp) as f32;
        self.position + self.velocity * elapsed.min(5.0)
    }

    /// Decay confidence over time.
    pub fn decay(&mut self, dt: f32) {
        self.confidence = (self.confidence - dt * 0.05).max(0.0);
    }

    /// Whether this entry is still relevant.
    pub fn is_relevant(&self, current_time: f64) -> bool {
        self.confidence > 0.05 && (current_time - self.timestamp) < 60.0
    }
}

// ---------------------------------------------------------------------------
// SearchPattern
// ---------------------------------------------------------------------------

/// A search behavior for AI looking for a lost target.
#[derive(Debug, Clone)]
pub struct SearchPattern {
    /// The searcher entity ID.
    pub searcher_id: u64,
    /// Target entity ID.
    pub target_id: u64,
    /// Search origin (last known position).
    pub origin: Vec3,
    /// Current search radius.
    pub radius: f32,
    /// Maximum search radius.
    pub max_radius: f32,
    /// Search duration remaining (seconds).
    pub duration: f32,
    /// Search points to visit.
    pub search_points: Vec<SearchPoint>,
    /// Current search point index.
    pub current_point: usize,
    /// Search pattern type.
    pub pattern_type: SearchPatternType,
    /// Whether the search is active.
    pub active: bool,
    /// Number of points visited.
    pub points_visited: u32,
}

/// A point to visit during a search.
#[derive(Debug, Clone)]
pub struct SearchPoint {
    /// Position to investigate.
    pub position: Vec3,
    /// How long to stay and look (seconds).
    pub look_time: f32,
    /// Whether this point has been checked.
    pub checked: bool,
    /// Priority (higher = check first).
    pub priority: f32,
}

/// Type of search pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPatternType {
    /// Check the last known position first, then expand outward.
    ExpandingCircle,
    /// Check in the direction the target was heading.
    PredictedPath,
    /// Check key positions (doorways, corners).
    KeyPoints,
    /// Random wandering near LKP.
    RandomWander,
    /// Systematic grid search.
    GridSearch,
}

impl SearchPattern {
    /// Create a new search pattern.
    pub fn new(
        searcher_id: u64,
        target_id: u64,
        origin: Vec3,
        pattern_type: SearchPatternType,
    ) -> Self {
        let mut pattern = Self {
            searcher_id,
            target_id,
            origin,
            radius: 5.0,
            max_radius: MAX_SEARCH_RADIUS,
            duration: DEFAULT_SEARCH_DURATION,
            search_points: Vec::new(),
            current_point: 0,
            pattern_type,
            active: true,
            points_visited: 0,
        };
        pattern.generate_points();
        pattern
    }

    /// Generate search points based on pattern type.
    pub fn generate_points(&mut self) {
        self.search_points.clear();

        match self.pattern_type {
            SearchPatternType::ExpandingCircle => {
                // Start at origin, then concentric circles
                self.search_points.push(SearchPoint {
                    position: self.origin,
                    look_time: 3.0,
                    checked: false,
                    priority: 10.0,
                });

                for ring in 1..=4 {
                    let r = self.radius * ring as f32;
                    let points_in_ring = (ring * 4).min(12);
                    for i in 0..points_in_ring {
                        let angle = (i as f32 / points_in_ring as f32)
                            * std::f32::consts::TAU;
                        let pos = Vec3::new(
                            self.origin.x + angle.cos() * r,
                            self.origin.y,
                            self.origin.z + angle.sin() * r,
                        );
                        self.search_points.push(SearchPoint {
                            position: pos,
                            look_time: 2.0,
                            checked: false,
                            priority: 10.0 - ring as f32,
                        });
                    }
                }
            }
            SearchPatternType::PredictedPath => {
                // Points along the predicted direction
                self.search_points.push(SearchPoint {
                    position: self.origin,
                    look_time: 2.0,
                    checked: false,
                    priority: 10.0,
                });
                // Without velocity info, use a forward sweep
                for dist in 1..=6 {
                    let d = dist as f32 * 5.0;
                    for angle in &[0.0f32, 0.5, -0.5, 1.0, -1.0] {
                        let pos = Vec3::new(
                            self.origin.x + angle.cos() * d,
                            self.origin.y,
                            self.origin.z + angle.sin() * d,
                        );
                        self.search_points.push(SearchPoint {
                            position: pos,
                            look_time: 1.5,
                            checked: false,
                            priority: 10.0 - dist as f32,
                        });
                    }
                }
            }
            _ => {
                // Default: check origin and 8 surrounding points
                self.search_points.push(SearchPoint {
                    position: self.origin,
                    look_time: 3.0,
                    checked: false,
                    priority: 10.0,
                });
                for angle_idx in 0..8 {
                    let angle = angle_idx as f32 * std::f32::consts::TAU / 8.0;
                    let pos = Vec3::new(
                        self.origin.x + angle.cos() * 10.0,
                        self.origin.y,
                        self.origin.z + angle.sin() * 10.0,
                    );
                    self.search_points.push(SearchPoint {
                        position: pos,
                        look_time: 2.0,
                        checked: false,
                        priority: 5.0,
                    });
                }
            }
        }

        // Sort by priority (highest first)
        self.search_points
            .sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Get the current point to search.
    pub fn current_search_point(&self) -> Option<&SearchPoint> {
        self.search_points.get(self.current_point)
    }

    /// Advance to the next search point.
    pub fn advance(&mut self) -> bool {
        if self.current_point < self.search_points.len() {
            if let Some(point) = self.search_points.get_mut(self.current_point) {
                point.checked = true;
            }
            self.points_visited += 1;
            self.current_point += 1;
        }
        self.current_point < self.search_points.len()
    }

    /// Update the search (decay duration, expand radius).
    pub fn update(&mut self, dt: f32) {
        self.duration -= dt;
        self.radius = (self.radius + SEARCH_EXPANSION_RATE * dt).min(self.max_radius);

        if self.duration <= 0.0 {
            self.active = false;
        }
    }

    /// Whether the search is complete (all points checked or timed out).
    pub fn is_complete(&self) -> bool {
        !self.active
            || self.current_point >= self.search_points.len()
    }

    /// Get the search completion percentage.
    pub fn completion_percentage(&self) -> f32 {
        if self.search_points.is_empty() {
            return 1.0;
        }
        self.points_visited as f32 / self.search_points.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Disguise
// ---------------------------------------------------------------------------

/// A disguise that reduces detection by specific factions or entity types.
#[derive(Debug, Clone)]
pub struct Disguise {
    /// Disguise unique ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Faction IDs that this disguise fools.
    pub fools_factions: Vec<u32>,
    /// Effectiveness (0..1, how much it reduces detection).
    pub effectiveness: f32,
    /// Durability (0..1, degrades with suspicious actions).
    pub durability: f32,
    /// Whether the disguise is currently active.
    pub active: bool,
    /// Actions that break the disguise.
    pub break_actions: Vec<DisguiseBreakAction>,
    /// Required items to maintain the disguise.
    pub required_items: Vec<String>,
}

/// Actions that can break a disguise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisguiseBreakAction {
    /// Attacking anyone.
    Attack,
    /// Entering a restricted area.
    RestrictedArea,
    /// Being seen by a specific (high-rank) NPC.
    SeenByOfficer,
    /// Running (unusual behavior).
    Running,
    /// Using prohibited equipment.
    ProhibitedEquipment,
    /// Getting too close to another NPC.
    TooClose,
    /// Performing a suspicious action.
    SuspiciousAction,
}

impl Disguise {
    /// Create a new disguise.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            fools_factions: Vec::new(),
            effectiveness: 0.8,
            durability: 1.0,
            active: false,
            break_actions: vec![
                DisguiseBreakAction::Attack,
                DisguiseBreakAction::ProhibitedEquipment,
            ],
            required_items: Vec::new(),
        }
    }

    /// Add a faction that this disguise fools.
    pub fn fools_faction(mut self, faction_id: u32) -> Self {
        self.fools_factions.push(faction_id);
        self
    }

    /// Set effectiveness.
    pub fn with_effectiveness(mut self, eff: f32) -> Self {
        self.effectiveness = eff.clamp(0.0, 1.0);
        self
    }

    /// Add a break action.
    pub fn breaks_on(mut self, action: DisguiseBreakAction) -> Self {
        if !self.break_actions.contains(&action) {
            self.break_actions.push(action);
        }
        self
    }

    /// Check if an action breaks this disguise.
    pub fn is_broken_by(&self, action: DisguiseBreakAction) -> bool {
        self.break_actions.contains(&action)
    }

    /// Activate the disguise.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate the disguise.
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Degrade durability. Returns false if disguise is destroyed.
    pub fn degrade(&mut self, amount: f32) -> bool {
        self.durability = (self.durability - amount).max(0.0);
        if self.durability <= 0.0 {
            self.active = false;
            false
        } else {
            true
        }
    }

    /// Check if this disguise fools a specific faction.
    pub fn fools(&self, faction_id: u32) -> bool {
        self.active && self.fools_factions.contains(&faction_id)
    }

    /// Get the effective detection reduction for a faction.
    pub fn detection_reduction(&self, faction_id: u32) -> f32 {
        if self.fools(faction_id) {
            self.effectiveness * self.durability
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// StealthComponent
// ---------------------------------------------------------------------------

/// Per-entity stealth state.
#[derive(Debug)]
pub struct StealthComponent {
    /// Entity ID.
    pub entity_id: u64,
    /// Current visibility (0 = invisible, 100 = fully visible).
    pub visibility: f32,
    /// Current noise output level.
    pub noise_level: f32,
    /// Current posture.
    pub posture: Posture,
    /// Current surface type being walked on.
    pub surface: SurfaceType,
    /// Movement speed (world units/second).
    pub movement_speed: f32,
    /// Light level at the entity's position (0..1).
    pub light_level: f32,
    /// Whether the entity is in shadow/cover.
    pub in_cover: bool,
    /// Active disguises.
    pub disguises: Vec<Disguise>,
    /// Last known positions tracked by AI.
    pub last_known_positions: VecDeque<LastKnownPositionEntry>,
    /// Alert states of watchers: watcher_id -> alert_state.
    pub watcher_states: HashMap<u64, AlertState>,
    /// Active search patterns from AI.
    pub active_searches: Vec<SearchPattern>,
    /// Whether the entity is invisible (magic/ability).
    pub invisible: bool,
    /// Invisibility effectiveness (0..1).
    pub invisibility_strength: f32,
    /// Detection reduction from abilities/items.
    pub detection_reduction: f32,
    /// Noise reduction from abilities/items.
    pub noise_reduction_bonus: f32,
}

impl StealthComponent {
    /// Create a new stealth component.
    pub fn new(entity_id: u64) -> Self {
        Self {
            entity_id,
            visibility: MAX_VISIBILITY,
            noise_level: 0.0,
            posture: Posture::Standing,
            surface: SurfaceType::Stone,
            movement_speed: 0.0,
            light_level: 1.0,
            in_cover: false,
            disguises: Vec::new(),
            last_known_positions: VecDeque::new(),
            watcher_states: HashMap::new(),
            active_searches: Vec::new(),
            invisible: false,
            invisibility_strength: 0.0,
            detection_reduction: 0.0,
            noise_reduction_bonus: 0.0,
        }
    }

    /// Calculate the effective visibility.
    pub fn calculate_visibility(&mut self) {
        if self.invisible {
            self.visibility = MAX_VISIBILITY * (1.0 - self.invisibility_strength);
            return;
        }

        let mut vis = MAX_VISIBILITY;

        // Light level: darker = less visible
        vis *= self.light_level;

        // Posture
        vis *= 1.0 - self.posture.visibility_reduction();

        // Movement: faster = more visible
        let movement_factor = 1.0 + self.movement_speed * MOVEMENT_VISIBILITY_FACTOR * 0.02;
        vis *= movement_factor.min(2.0);

        // Cover
        if self.in_cover {
            vis *= 0.5;
        }

        // Ability-based reduction
        vis *= 1.0 - self.detection_reduction.clamp(0.0, 0.9);

        self.visibility = vis.clamp(MIN_VISIBILITY, MAX_VISIBILITY);
    }

    /// Calculate the noise output.
    pub fn calculate_noise(&mut self) {
        if self.movement_speed < 0.1 {
            // Standing still — no footstep noise
            self.noise_level = 0.0;
            return;
        }

        let base = self.movement_speed * 5.0;
        let surface_mult = self.surface.noise_multiplier();
        let posture_reduction = 1.0 - self.posture.noise_reduction();
        let bonus_reduction = 1.0 - self.noise_reduction_bonus.clamp(0.0, 0.8);

        self.noise_level = (base * surface_mult * posture_reduction * bonus_reduction)
            .clamp(0.0, MAX_NOISE);
    }

    /// Update visibility and noise calculations.
    pub fn update(&mut self, dt: f32) {
        self.calculate_visibility();
        self.calculate_noise();

        // Decay LKP confidence
        for lkp in &mut self.last_known_positions {
            lkp.decay(dt);
        }

        // Update active searches
        for search in &mut self.active_searches {
            search.update(dt);
        }
        self.active_searches.retain(|s| s.active);
    }

    /// Add a last-known-position entry.
    pub fn add_lkp(&mut self, target_id: u64, position: Vec3, velocity: Vec3, timestamp: f64) {
        // Update existing or add new
        if let Some(existing) = self
            .last_known_positions
            .iter_mut()
            .find(|l| l.target_id == target_id)
        {
            existing.position = position;
            existing.velocity = velocity;
            existing.timestamp = timestamp;
            existing.confidence = 1.0;
            existing.searched = false;
        } else {
            if self.last_known_positions.len() >= MAX_LKP_HISTORY {
                self.last_known_positions.pop_front();
            }
            self.last_known_positions.push_back(
                LastKnownPositionEntry::new(target_id, position, velocity, timestamp),
            );
        }
    }

    /// Get the most recent LKP for a target.
    pub fn get_lkp(&self, target_id: u64) -> Option<&LastKnownPositionEntry> {
        self.last_known_positions
            .iter()
            .rev()
            .find(|l| l.target_id == target_id)
    }

    /// Set the alert state from a specific watcher.
    pub fn set_watcher_alert(&mut self, watcher_id: u64, state: AlertState) {
        self.watcher_states.insert(watcher_id, state);
    }

    /// Get the highest alert state from any watcher.
    pub fn highest_alert(&self) -> AlertState {
        self.watcher_states
            .values()
            .max_by_key(|s| match s {
                AlertState::Unaware => 0,
                AlertState::Suspicious => 1,
                AlertState::Searching => 2,
                AlertState::Alert => 3,
                AlertState::Engaged => 4,
            })
            .copied()
            .unwrap_or(AlertState::Unaware)
    }

    /// Check if visible to a specific detector at a given distance.
    pub fn visible_to(
        &self,
        detector_alert: AlertState,
        distance: f32,
        max_range: f32,
        detector_faction: Option<u32>,
    ) -> bool {
        let effective_vis = self.effective_visibility(detector_faction);
        let threshold = DEFAULT_DETECTION_THRESHOLD / detector_alert.sensitivity_multiplier();
        let range_factor = (1.0 - distance / max_range).max(0.0);

        effective_vis * range_factor > threshold
    }

    /// Get effective visibility considering disguises.
    pub fn effective_visibility(&self, observer_faction: Option<u32>) -> f32 {
        let mut vis = self.visibility;

        if let Some(faction_id) = observer_faction {
            for disguise in &self.disguises {
                let reduction = disguise.detection_reduction(faction_id);
                vis *= 1.0 - reduction;
            }
        }

        vis
    }

    /// Equip a disguise.
    pub fn equip_disguise(&mut self, mut disguise: Disguise) -> bool {
        if self.disguises.len() >= MAX_DISGUISES {
            return false;
        }
        disguise.activate();
        self.disguises.push(disguise);
        true
    }

    /// Remove a disguise by ID.
    pub fn remove_disguise(&mut self, disguise_id: &str) -> bool {
        let before = self.disguises.len();
        self.disguises.retain(|d| d.id != disguise_id);
        self.disguises.len() < before
    }

    /// Check and break disguises for a given action.
    pub fn check_disguise_break(&mut self, action: DisguiseBreakAction) {
        for disguise in &mut self.disguises {
            if disguise.is_broken_by(action) {
                disguise.deactivate();
            }
        }
    }

    /// Start a search pattern.
    pub fn start_search(
        &mut self,
        searcher_id: u64,
        target_id: u64,
        origin: Vec3,
        pattern: SearchPatternType,
    ) {
        let search = SearchPattern::new(searcher_id, target_id, origin, pattern);
        self.active_searches.push(search);
    }

    /// Check if any search is active for a target.
    pub fn is_searching_for(&self, target_id: u64) -> bool {
        self.active_searches
            .iter()
            .any(|s| s.target_id == target_id && s.active)
    }
}

// ---------------------------------------------------------------------------
// StealthEvent
// ---------------------------------------------------------------------------

/// Events emitted by the stealth system.
#[derive(Debug, Clone)]
pub enum StealthEvent {
    /// An entity was detected.
    Detected {
        target: u64,
        detector: u64,
        visibility: f32,
    },
    /// An entity lost detection.
    LostSight {
        target: u64,
        detector: u64,
        last_position: Vec3,
    },
    /// Alert state changed.
    AlertChanged {
        detector: u64,
        target: u64,
        old_state: AlertState,
        new_state: AlertState,
    },
    /// A noise was heard.
    NoiseHeard {
        listener: u64,
        noise_source: u64,
        noise_type: NoiseType,
        volume: f32,
    },
    /// A disguise was broken.
    DisguiseBroken {
        entity: u64,
        disguise_id: String,
        action: DisguiseBreakAction,
    },
    /// A search was started.
    SearchStarted {
        searcher: u64,
        target: u64,
        origin: Vec3,
    },
    /// A search completed without finding the target.
    SearchFailed {
        searcher: u64,
        target: u64,
    },
}

// ---------------------------------------------------------------------------
// StealthSystem
// ---------------------------------------------------------------------------

/// Top-level stealth system manager.
pub struct StealthSystem {
    /// Per-entity stealth components.
    components: HashMap<u64, StealthComponent>,
    /// Pending noise events this tick.
    noise_events: Vec<NoiseEvent>,
    /// Emitted events.
    events: Vec<StealthEvent>,
    /// Light probe callback placeholder: entity_id -> light_level.
    light_levels: HashMap<u64, f32>,
}

impl StealthSystem {
    /// Create a new stealth system.
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            noise_events: Vec::new(),
            events: Vec::new(),
            light_levels: HashMap::new(),
        }
    }

    /// Register an entity for stealth tracking.
    pub fn register(&mut self, entity_id: u64) {
        if self.components.len() < MAX_STEALTH_ENTITIES {
            self.components
                .entry(entity_id)
                .or_insert_with(|| StealthComponent::new(entity_id));
        }
    }

    /// Unregister an entity.
    pub fn unregister(&mut self, entity_id: u64) {
        self.components.remove(&entity_id);
    }

    /// Get a stealth component.
    pub fn get(&self, entity_id: u64) -> Option<&StealthComponent> {
        self.components.get(&entity_id)
    }

    /// Get a mutable stealth component.
    pub fn get_mut(&mut self, entity_id: u64) -> Option<&mut StealthComponent> {
        self.components.get_mut(&entity_id)
    }

    /// Set the light level for an entity's position.
    pub fn set_light_level(&mut self, entity_id: u64, level: f32) {
        self.light_levels.insert(entity_id, level.clamp(0.0, 1.0));
    }

    /// Emit a noise event.
    pub fn emit_noise(&mut self, event: NoiseEvent) {
        if self.noise_events.len() < MAX_NOISE_EVENTS_PER_TICK {
            self.noise_events.push(event);
        }
    }

    /// Update all stealth components.
    pub fn update(&mut self, dt: f32) {
        // Apply light levels
        for (&entity_id, &level) in &self.light_levels {
            if let Some(comp) = self.components.get_mut(&entity_id) {
                comp.light_level = level;
            }
        }

        // Update components
        for comp in self.components.values_mut() {
            comp.update(dt);
        }

        // Process noise events
        let noise_events = std::mem::take(&mut self.noise_events);
        for noise in &noise_events {
            // Check each component to see if they can hear this noise
            let entity_ids: Vec<u64> = self.components.keys().copied().collect();
            for listener_id in entity_ids {
                if listener_id == noise.source {
                    continue;
                }
                if let Some(comp) = self.components.get(&listener_id) {
                    let volume = noise.volume_at(comp.last_known_positions
                        .back()
                        .map(|l| l.position)
                        .unwrap_or(Vec3::ZERO));

                    if volume > NOISE_AWARENESS_THRESHOLD {
                        self.events.push(StealthEvent::NoiseHeard {
                            listener: listener_id,
                            noise_source: noise.source,
                            noise_type: noise.noise_type,
                            volume,
                        });
                    }
                }
            }
        }
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<StealthEvent> {
        std::mem::take(&mut self.events)
    }

    /// Get entity count.
    pub fn entity_count(&self) -> usize {
        self.components.len()
    }
}

impl Default for StealthSystem {
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
    fn test_visibility_calculation() {
        let mut comp = StealthComponent::new(1);
        comp.light_level = 0.2;
        comp.posture = Posture::Crouching;
        comp.movement_speed = 0.0;
        comp.calculate_visibility();

        // Should be quite low: dark + crouching + still
        assert!(comp.visibility < 30.0);
    }

    #[test]
    fn test_noise_calculation() {
        let mut comp = StealthComponent::new(1);
        comp.movement_speed = 5.0;
        comp.surface = SurfaceType::Metal;
        comp.posture = Posture::Standing;
        comp.calculate_noise();

        let standing_noise = comp.noise_level;

        comp.posture = Posture::Crouching;
        comp.calculate_noise();
        let crouching_noise = comp.noise_level;

        assert!(crouching_noise < standing_noise);
    }

    #[test]
    fn test_alert_states() {
        let mut comp = StealthComponent::new(1);
        comp.set_watcher_alert(10, AlertState::Suspicious);
        comp.set_watcher_alert(11, AlertState::Alert);

        assert_eq!(comp.highest_alert(), AlertState::Alert);
    }

    #[test]
    fn test_disguise() {
        let disguise = Disguise::new("guard_uniform", "Guard Uniform")
            .fools_faction(1)
            .fools_faction(2)
            .with_effectiveness(0.9);

        let mut comp = StealthComponent::new(1);
        comp.equip_disguise(disguise);

        let vis = comp.effective_visibility(Some(1));
        assert!(vis < comp.visibility);

        // Breaking
        comp.check_disguise_break(DisguiseBreakAction::Attack);
        let vis_after = comp.effective_visibility(Some(1));
        assert!((vis_after - comp.visibility).abs() < 0.01);
    }

    #[test]
    fn test_noise_event() {
        let event = NoiseEvent::new(1, Vec3::new(10.0, 0.0, 10.0), NoiseType::Gunshot, 0.0);
        assert!(event.audible_at(Vec3::new(50.0, 0.0, 10.0)));
        assert!(!event.audible_at(Vec3::new(200.0, 0.0, 10.0)));
    }

    #[test]
    fn test_search_pattern() {
        let mut search = SearchPattern::new(
            10, 1,
            Vec3::new(50.0, 0.0, 50.0),
            SearchPatternType::ExpandingCircle,
        );

        assert!(!search.search_points.is_empty());
        assert!(search.advance());
        assert_eq!(search.points_visited, 1);
    }

    #[test]
    fn test_surface_noise() {
        assert!(SurfaceType::Metal.noise_multiplier() > SurfaceType::Carpet.noise_multiplier());
        assert!(SurfaceType::Snow.leaves_tracks());
        assert!(!SurfaceType::Stone.leaves_tracks());
    }
}
