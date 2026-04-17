//! Faction and reputation system.
//!
//! Manages factions, per-entity reputation with each faction, reputation
//! thresholds that determine standing (hostile to allied), inter-faction
//! relationships, reputation events, and faction territory control.
//!
//! # Key concepts
//!
//! - **Faction**: A named group with members, territory, and policies.
//! - **ReputationStanding**: Discrete standing levels from Hostile to Allied.
//! - **ReputationEvent**: A change to reputation triggered by in-game actions.
//! - **FactionRelationship**: How two factions feel about each other.
//! - **FactionTerritory**: Areas of the world controlled by a faction.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of factions.
pub const MAX_FACTIONS: usize = 64;

/// Maximum reputation value.
pub const MAX_REPUTATION: f32 = 100.0;

/// Minimum reputation value.
pub const MIN_REPUTATION: f32 = -100.0;

/// Default starting reputation with a neutral faction.
pub const DEFAULT_REPUTATION: f32 = 0.0;

/// Threshold for Hostile standing.
pub const HOSTILE_THRESHOLD: f32 = -60.0;

/// Threshold for Unfriendly standing.
pub const UNFRIENDLY_THRESHOLD: f32 = -20.0;

/// Threshold for Friendly standing.
pub const FRIENDLY_THRESHOLD: f32 = 20.0;

/// Threshold for Allied standing.
pub const ALLIED_THRESHOLD: f32 = 60.0;

/// Maximum reputation events in history.
pub const MAX_REPUTATION_HISTORY: usize = 256;

/// Maximum members per faction.
pub const MAX_FACTION_MEMBERS: usize = 4096;

/// Maximum territories per faction.
pub const MAX_TERRITORIES: usize = 64;

/// Reputation decay rate per game-hour (toward default).
pub const REPUTATION_DECAY_RATE: f32 = 0.1;

/// Reputation splash range (faction-to-faction influence on rep changes).
pub const REPUTATION_SPLASH_FACTOR: f32 = 0.25;

// ---------------------------------------------------------------------------
// ReputationStanding
// ---------------------------------------------------------------------------

/// Discrete reputation standing with a faction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ReputationStanding {
    /// Will attack on sight, refuses all interaction.
    Hostile,
    /// Refuses trade and most interactions. May attack if provoked.
    Unfriendly,
    /// Default standing. Standard interactions available.
    Neutral,
    /// Positive standing. Better prices, more quests available.
    Friendly,
    /// Maximum standing. Full access to faction resources and quests.
    Allied,
}

impl ReputationStanding {
    /// Convert a numeric reputation value to a standing.
    pub fn from_value(value: f32) -> Self {
        if value <= HOSTILE_THRESHOLD {
            Self::Hostile
        } else if value <= UNFRIENDLY_THRESHOLD {
            Self::Unfriendly
        } else if value >= ALLIED_THRESHOLD {
            Self::Allied
        } else if value >= FRIENDLY_THRESHOLD {
            Self::Friendly
        } else {
            Self::Neutral
        }
    }

    /// Get the display label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Hostile => "Hostile",
            Self::Unfriendly => "Unfriendly",
            Self::Neutral => "Neutral",
            Self::Friendly => "Friendly",
            Self::Allied => "Allied",
        }
    }

    /// Color hex for UI display.
    pub fn color_hex(&self) -> &'static str {
        match self {
            Self::Hostile => "#FF0000",
            Self::Unfriendly => "#FF8800",
            Self::Neutral => "#FFFF00",
            Self::Friendly => "#00FF00",
            Self::Allied => "#00FFFF",
        }
    }

    /// Whether this standing allows trade.
    pub fn allows_trade(&self) -> bool {
        matches!(self, Self::Neutral | Self::Friendly | Self::Allied)
    }

    /// Whether this standing allows quest access.
    pub fn allows_quests(&self) -> bool {
        matches!(self, Self::Friendly | Self::Allied)
    }

    /// Whether this standing means the faction will attack.
    pub fn will_attack(&self) -> bool {
        matches!(self, Self::Hostile)
    }

    /// Price discount multiplier based on standing.
    pub fn price_multiplier(&self) -> f32 {
        match self {
            Self::Hostile => 3.0,
            Self::Unfriendly => 1.5,
            Self::Neutral => 1.0,
            Self::Friendly => 0.9,
            Self::Allied => 0.75,
        }
    }

    /// Minimum reputation value for this standing.
    pub fn min_value(&self) -> f32 {
        match self {
            Self::Hostile => MIN_REPUTATION,
            Self::Unfriendly => HOSTILE_THRESHOLD + 0.01,
            Self::Neutral => UNFRIENDLY_THRESHOLD + 0.01,
            Self::Friendly => FRIENDLY_THRESHOLD,
            Self::Allied => ALLIED_THRESHOLD,
        }
    }
}

// ---------------------------------------------------------------------------
// FactionRelationshipType
// ---------------------------------------------------------------------------

/// How two factions relate to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FactionRelationshipType {
    /// Factions actively fight each other.
    Enemy,
    /// Factions distrust each other but don't fight.
    Rival,
    /// No special relationship.
    Neutral,
    /// Factions cooperate when convenient.
    Friendly,
    /// Factions share goals and defend each other.
    Ally,
}

impl FactionRelationshipType {
    /// Reputation splash multiplier (how much rep changes splash to related factions).
    pub fn splash_multiplier(&self) -> f32 {
        match self {
            Self::Enemy => -0.5,
            Self::Rival => -0.2,
            Self::Neutral => 0.0,
            Self::Friendly => 0.3,
            Self::Ally => 0.5,
        }
    }

    /// Whether members of these factions should fight.
    pub fn is_hostile(&self) -> bool {
        matches!(self, Self::Enemy)
    }

    /// Display label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Enemy => "Enemy",
            Self::Rival => "Rival",
            Self::Neutral => "Neutral",
            Self::Friendly => "Friendly",
            Self::Ally => "Ally",
        }
    }
}

// ---------------------------------------------------------------------------
// FactionRelationship
// ---------------------------------------------------------------------------

/// Relationship between two specific factions.
#[derive(Debug, Clone)]
pub struct FactionRelationship {
    /// First faction ID.
    pub faction_a: u32,
    /// Second faction ID.
    pub faction_b: u32,
    /// Relationship type.
    pub relationship: FactionRelationshipType,
    /// Numeric relationship value (-100..100).
    pub value: f32,
    /// Whether this relationship is locked (won't change from events).
    pub locked: bool,
    /// History of changes.
    pub history: Vec<String>,
}

impl FactionRelationship {
    /// Create a new faction relationship.
    pub fn new(
        faction_a: u32,
        faction_b: u32,
        relationship: FactionRelationshipType,
    ) -> Self {
        let value = match relationship {
            FactionRelationshipType::Enemy => -80.0,
            FactionRelationshipType::Rival => -30.0,
            FactionRelationshipType::Neutral => 0.0,
            FactionRelationshipType::Friendly => 40.0,
            FactionRelationshipType::Ally => 80.0,
        };

        Self {
            faction_a,
            faction_b,
            relationship,
            value,
            locked: false,
            history: Vec::new(),
        }
    }

    /// Modify the relationship value.
    pub fn modify(&mut self, delta: f32, reason: impl Into<String>) {
        if self.locked {
            return;
        }
        self.value = (self.value + delta).clamp(MIN_REPUTATION, MAX_REPUTATION);
        self.history.push(reason.into());

        // Update relationship type based on value
        self.relationship = if self.value <= -50.0 {
            FactionRelationshipType::Enemy
        } else if self.value <= -15.0 {
            FactionRelationshipType::Rival
        } else if self.value >= 50.0 {
            FactionRelationshipType::Ally
        } else if self.value >= 15.0 {
            FactionRelationshipType::Friendly
        } else {
            FactionRelationshipType::Neutral
        };
    }

    /// Check if this relationship involves a given faction.
    pub fn involves(&self, faction_id: u32) -> bool {
        self.faction_a == faction_id || self.faction_b == faction_id
    }

    /// Get the other faction in the relationship.
    pub fn other_faction(&self, faction_id: u32) -> Option<u32> {
        if self.faction_a == faction_id {
            Some(self.faction_b)
        } else if self.faction_b == faction_id {
            Some(self.faction_a)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// FactionTerritory
// ---------------------------------------------------------------------------

/// An area of the world controlled by a faction.
#[derive(Debug, Clone)]
pub struct FactionTerritory {
    /// Territory unique ID.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Center position (x, z in world coordinates).
    pub center: (f32, f32),
    /// Radius of control.
    pub radius: f32,
    /// Control strength (0..1).
    pub control_strength: f32,
    /// Whether the territory is contested.
    pub contested: bool,
    /// Faction currently contesting (if any).
    pub contested_by: Option<u32>,
    /// Resources generated by this territory.
    pub resources: HashMap<String, f32>,
    /// Important structures in the territory.
    pub structures: Vec<String>,
}

impl FactionTerritory {
    /// Create a new territory.
    pub fn new(
        id: u32,
        name: impl Into<String>,
        center: (f32, f32),
        radius: f32,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            center,
            radius,
            control_strength: 1.0,
            contested: false,
            contested_by: None,
            resources: HashMap::new(),
            structures: Vec::new(),
        }
    }

    /// Check if a point is within this territory.
    pub fn contains(&self, x: f32, z: f32) -> bool {
        let dx = x - self.center.0;
        let dz = z - self.center.1;
        (dx * dx + dz * dz).sqrt() <= self.radius
    }

    /// Contest this territory.
    pub fn contest(&mut self, attacker: u32) {
        self.contested = true;
        self.contested_by = Some(attacker);
        self.control_strength = (self.control_strength - 0.1).max(0.0);
    }

    /// Reinforce this territory.
    pub fn reinforce(&mut self, amount: f32) {
        self.control_strength = (self.control_strength + amount).min(1.0);
        if self.control_strength >= 0.8 {
            self.contested = false;
            self.contested_by = None;
        }
    }

    /// Add a resource to this territory.
    pub fn add_resource(&mut self, resource: impl Into<String>, amount: f32) {
        let resource = resource.into();
        *self.resources.entry(resource).or_insert(0.0) += amount;
    }
}

// ---------------------------------------------------------------------------
// ReputationEvent
// ---------------------------------------------------------------------------

/// An event that changes reputation.
#[derive(Debug, Clone)]
pub struct ReputationEvent {
    /// Unique event ID.
    pub id: u64,
    /// The entity whose reputation changed.
    pub entity_id: u64,
    /// The faction affected.
    pub faction_id: u32,
    /// Amount of reputation change.
    pub delta: f32,
    /// Reason for the change.
    pub reason: ReputationReason,
    /// Description.
    pub description: String,
    /// Game time when the event occurred.
    pub timestamp: f64,
    /// Whether this event has been processed.
    pub processed: bool,
}

/// Reason for a reputation change.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReputationReason {
    /// Killed a member of the faction.
    KilledMember,
    /// Helped a member of the faction.
    HelpedMember,
    /// Completed a quest for the faction.
    CompletedQuest,
    /// Failed a quest for the faction.
    FailedQuest,
    /// Donated to the faction.
    Donation,
    /// Attacked a member without killing.
    AttackedMember,
    /// Stole from the faction.
    Theft,
    /// Trespassed in faction territory.
    Trespass,
    /// Completed a trade deal.
    Trade,
    /// Discovered information for the faction.
    Intelligence,
    /// Betrayed the faction.
    Betrayal,
    /// Diplomatic action.
    Diplomacy,
    /// Custom reason.
    Custom(String),
}

impl ReputationReason {
    /// Default reputation change for this reason.
    pub fn default_delta(&self) -> f32 {
        match self {
            Self::KilledMember => -15.0,
            Self::HelpedMember => 5.0,
            Self::CompletedQuest => 10.0,
            Self::FailedQuest => -5.0,
            Self::Donation => 3.0,
            Self::AttackedMember => -8.0,
            Self::Theft => -10.0,
            Self::Trespass => -3.0,
            Self::Trade => 2.0,
            Self::Intelligence => 8.0,
            Self::Betrayal => -25.0,
            Self::Diplomacy => 5.0,
            Self::Custom(_) => 0.0,
        }
    }

    /// Description of this reason.
    pub fn description(&self) -> String {
        match self {
            Self::KilledMember => "Killed a faction member".into(),
            Self::HelpedMember => "Helped a faction member".into(),
            Self::CompletedQuest => "Completed a faction quest".into(),
            Self::FailedQuest => "Failed a faction quest".into(),
            Self::Donation => "Made a donation to the faction".into(),
            Self::AttackedMember => "Attacked a faction member".into(),
            Self::Theft => "Stole from the faction".into(),
            Self::Trespass => "Trespassed in faction territory".into(),
            Self::Trade => "Completed a trade deal".into(),
            Self::Intelligence => "Provided intelligence".into(),
            Self::Betrayal => "Betrayed the faction".into(),
            Self::Diplomacy => "Diplomatic action".into(),
            Self::Custom(s) => s.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// FactionPolicy
// ---------------------------------------------------------------------------

/// Policies that govern faction behavior.
#[derive(Debug, Clone)]
pub struct FactionPolicy {
    /// Whether the faction attacks hostiles on sight.
    pub attack_on_sight: bool,
    /// Whether the faction accepts surrenders.
    pub accepts_surrender: bool,
    /// Whether the faction trades with neutrals.
    pub trades_with_neutrals: bool,
    /// Whether the faction shares territory information.
    pub shares_intel: bool,
    /// Tax rate on territory resources.
    pub tax_rate: f32,
    /// Minimum reputation to join the faction.
    pub join_threshold: f32,
    /// Whether deserters are hunted.
    pub hunts_deserters: bool,
}

impl Default for FactionPolicy {
    fn default() -> Self {
        Self {
            attack_on_sight: true,
            accepts_surrender: true,
            trades_with_neutrals: true,
            shares_intel: false,
            tax_rate: 0.1,
            join_threshold: 20.0,
            hunts_deserters: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Faction
// ---------------------------------------------------------------------------

/// A faction or group in the game world.
#[derive(Debug, Clone)]
pub struct Faction {
    /// Unique faction ID.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Icon/emblem identifier.
    pub icon: String,
    /// Color for map display.
    pub color: (u8, u8, u8),
    /// Leader entity ID.
    pub leader: Option<u64>,
    /// Member entity IDs.
    members: HashSet<u64>,
    /// Faction policies.
    pub policy: FactionPolicy,
    /// Territories controlled.
    territories: Vec<FactionTerritory>,
    /// Whether the faction is active.
    pub active: bool,
    /// Faction resources (shared pool).
    pub resources: HashMap<String, f64>,
    /// Faction rank names (lowest to highest).
    pub ranks: Vec<String>,
    /// Member ranks.
    member_ranks: HashMap<u64, usize>,
}

impl Faction {
    /// Create a new faction.
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: String::new(),
            icon: String::new(),
            color: (128, 128, 128),
            leader: None,
            members: HashSet::new(),
            policy: FactionPolicy::default(),
            territories: Vec::new(),
            active: true,
            resources: HashMap::new(),
            ranks: vec![
                "Initiate".into(),
                "Member".into(),
                "Veteran".into(),
                "Officer".into(),
                "Commander".into(),
            ],
            member_ranks: HashMap::new(),
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the color.
    pub fn with_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.color = (r, g, b);
        self
    }

    /// Set the leader.
    pub fn with_leader(mut self, leader: u64) -> Self {
        self.leader = Some(leader);
        self
    }

    /// Add a member to the faction.
    pub fn add_member(&mut self, entity_id: u64) -> bool {
        if self.members.len() >= MAX_FACTION_MEMBERS {
            return false;
        }
        if self.members.insert(entity_id) {
            self.member_ranks.insert(entity_id, 0);
            true
        } else {
            false
        }
    }

    /// Remove a member.
    pub fn remove_member(&mut self, entity_id: u64) -> bool {
        self.member_ranks.remove(&entity_id);
        self.members.remove(&entity_id)
    }

    /// Check if an entity is a member.
    pub fn is_member(&self, entity_id: u64) -> bool {
        self.members.contains(&entity_id)
    }

    /// Get member count.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Get all members.
    pub fn members(&self) -> &HashSet<u64> {
        &self.members
    }

    /// Set a member's rank.
    pub fn set_rank(&mut self, entity_id: u64, rank: usize) {
        if self.members.contains(&entity_id) && rank < self.ranks.len() {
            self.member_ranks.insert(entity_id, rank);
        }
    }

    /// Get a member's rank.
    pub fn get_rank(&self, entity_id: u64) -> Option<usize> {
        self.member_ranks.get(&entity_id).copied()
    }

    /// Get a member's rank name.
    pub fn get_rank_name(&self, entity_id: u64) -> Option<&str> {
        self.member_ranks
            .get(&entity_id)
            .and_then(|&r| self.ranks.get(r))
            .map(|s| s.as_str())
    }

    /// Promote a member (increase rank).
    pub fn promote(&mut self, entity_id: u64) -> bool {
        if let Some(rank) = self.member_ranks.get_mut(&entity_id) {
            if *rank + 1 < self.ranks.len() {
                *rank += 1;
                return true;
            }
        }
        false
    }

    /// Demote a member (decrease rank).
    pub fn demote(&mut self, entity_id: u64) -> bool {
        if let Some(rank) = self.member_ranks.get_mut(&entity_id) {
            if *rank > 0 {
                *rank -= 1;
                return true;
            }
        }
        false
    }

    /// Add a territory.
    pub fn add_territory(&mut self, territory: FactionTerritory) -> bool {
        if self.territories.len() >= MAX_TERRITORIES {
            return false;
        }
        self.territories.push(territory);
        true
    }

    /// Remove a territory by ID.
    pub fn remove_territory(&mut self, territory_id: u32) -> bool {
        let before = self.territories.len();
        self.territories.retain(|t| t.id != territory_id);
        self.territories.len() < before
    }

    /// Get territories.
    pub fn territories(&self) -> &[FactionTerritory] {
        &self.territories
    }

    /// Get mutable territories.
    pub fn territories_mut(&mut self) -> &mut Vec<FactionTerritory> {
        &mut self.territories
    }

    /// Check if a position is within any faction territory.
    pub fn is_in_territory(&self, x: f32, z: f32) -> bool {
        self.territories.iter().any(|t| t.contains(x, z))
    }

    /// Find which territory a position is in.
    pub fn territory_at(&self, x: f32, z: f32) -> Option<&FactionTerritory> {
        self.territories.iter().find(|t| t.contains(x, z))
    }

    /// Add a resource to the faction pool.
    pub fn add_resource(&mut self, resource: impl Into<String>, amount: f64) {
        *self.resources.entry(resource.into()).or_insert(0.0) += amount;
    }

    /// Spend a resource. Returns true if successful.
    pub fn spend_resource(&mut self, resource: &str, amount: f64) -> bool {
        if let Some(current) = self.resources.get_mut(resource) {
            if *current >= amount {
                *current -= amount;
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// FactionSystem
// ---------------------------------------------------------------------------

/// Top-level system managing all factions and reputation.
pub struct FactionSystem {
    /// All factions.
    factions: HashMap<u32, Faction>,
    /// Per-entity reputation with each faction: entity -> (faction_id -> reputation).
    reputation: HashMap<u64, HashMap<u32, f32>>,
    /// Inter-faction relationships.
    relationships: Vec<FactionRelationship>,
    /// Reputation event queue.
    pending_events: VecDeque<ReputationEvent>,
    /// Processed events history.
    event_history: VecDeque<ReputationEvent>,
    /// Next event ID.
    next_event_id: u64,
    /// Next faction ID.
    next_faction_id: u32,
    /// Whether reputation decay is enabled.
    pub decay_enabled: bool,
}

impl FactionSystem {
    /// Create a new faction system.
    pub fn new() -> Self {
        Self {
            factions: HashMap::new(),
            reputation: HashMap::new(),
            relationships: Vec::new(),
            pending_events: VecDeque::new(),
            event_history: VecDeque::new(),
            next_event_id: 1,
            next_faction_id: 1,
            decay_enabled: false,
        }
    }

    /// Register a new faction. Returns the faction ID.
    pub fn register_faction(&mut self, name: impl Into<String>) -> u32 {
        let id = self.next_faction_id;
        self.next_faction_id += 1;
        let faction = Faction::new(id, name);
        self.factions.insert(id, faction);
        id
    }

    /// Register a faction with a predefined ID.
    pub fn register_faction_with_id(&mut self, id: u32, name: impl Into<String>) {
        let faction = Faction::new(id, name);
        self.factions.insert(id, faction);
        if id >= self.next_faction_id {
            self.next_faction_id = id + 1;
        }
    }

    /// Get a faction by ID.
    pub fn get_faction(&self, id: u32) -> Option<&Faction> {
        self.factions.get(&id)
    }

    /// Get a mutable faction by ID.
    pub fn get_faction_mut(&mut self, id: u32) -> Option<&mut Faction> {
        self.factions.get_mut(&id)
    }

    /// Remove a faction.
    pub fn remove_faction(&mut self, id: u32) -> bool {
        if self.factions.remove(&id).is_some() {
            self.relationships.retain(|r| !r.involves(id));
            // Clean up reputation entries
            for rep_map in self.reputation.values_mut() {
                rep_map.remove(&id);
            }
            true
        } else {
            false
        }
    }

    /// Get all faction IDs.
    pub fn faction_ids(&self) -> Vec<u32> {
        self.factions.keys().copied().collect()
    }

    /// Get the faction count.
    pub fn faction_count(&self) -> usize {
        self.factions.len()
    }

    // -----------------------------------------------------------------------
    // Reputation
    // -----------------------------------------------------------------------

    /// Get an entity's reputation with a faction.
    pub fn get_reputation(&self, entity_id: u64, faction_id: u32) -> f32 {
        self.reputation
            .get(&entity_id)
            .and_then(|m| m.get(&faction_id))
            .copied()
            .unwrap_or(DEFAULT_REPUTATION)
    }

    /// Set an entity's reputation with a faction directly.
    pub fn set_reputation(&mut self, entity_id: u64, faction_id: u32, value: f32) {
        let value = value.clamp(MIN_REPUTATION, MAX_REPUTATION);
        self.reputation
            .entry(entity_id)
            .or_insert_with(HashMap::new)
            .insert(faction_id, value);
    }

    /// Get an entity's standing with a faction.
    pub fn get_standing(&self, entity_id: u64, faction_id: u32) -> ReputationStanding {
        ReputationStanding::from_value(self.get_reputation(entity_id, faction_id))
    }

    /// Modify reputation and emit an event.
    pub fn modify_reputation(
        &mut self,
        entity_id: u64,
        faction_id: u32,
        delta: f32,
        reason: ReputationReason,
        game_time: f64,
    ) {
        let current = self.get_reputation(entity_id, faction_id);
        let new_value = (current + delta).clamp(MIN_REPUTATION, MAX_REPUTATION);
        self.set_reputation(entity_id, faction_id, new_value);

        let event = ReputationEvent {
            id: self.next_event_id,
            entity_id,
            faction_id,
            delta,
            reason: reason.clone(),
            description: reason.description(),
            timestamp: game_time,
            processed: false,
        };
        self.next_event_id += 1;
        self.pending_events.push_back(event);

        // Splash reputation to related factions
        self.apply_reputation_splash(entity_id, faction_id, delta);
    }

    /// Apply reputation changes to factions related to the modified faction.
    fn apply_reputation_splash(&mut self, entity_id: u64, faction_id: u32, delta: f32) {
        let splashes: Vec<(u32, f32)> = self
            .relationships
            .iter()
            .filter(|r| r.involves(faction_id))
            .filter_map(|r| {
                let other = r.other_faction(faction_id)?;
                let splash = delta * r.relationship.splash_multiplier() * REPUTATION_SPLASH_FACTOR;
                if splash.abs() > 0.01 {
                    Some((other, splash))
                } else {
                    None
                }
            })
            .collect();

        for (other_faction, splash_delta) in splashes {
            let current = self.get_reputation(entity_id, other_faction);
            let new_value = (current + splash_delta).clamp(MIN_REPUTATION, MAX_REPUTATION);
            self.set_reputation(entity_id, other_faction, new_value);
        }
    }

    /// Process a reputation reason using default deltas.
    pub fn apply_reputation_event(
        &mut self,
        entity_id: u64,
        faction_id: u32,
        reason: ReputationReason,
        game_time: f64,
    ) {
        let delta = reason.default_delta();
        self.modify_reputation(entity_id, faction_id, delta, reason, game_time);
    }

    /// Get all reputations for an entity.
    pub fn get_all_reputations(&self, entity_id: u64) -> HashMap<u32, (f32, ReputationStanding)> {
        let mut result = HashMap::new();
        if let Some(rep_map) = self.reputation.get(&entity_id) {
            for (&faction_id, &value) in rep_map {
                result.insert(faction_id, (value, ReputationStanding::from_value(value)));
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Faction relationships
    // -----------------------------------------------------------------------

    /// Set the relationship between two factions.
    pub fn set_relationship(
        &mut self,
        faction_a: u32,
        faction_b: u32,
        relationship: FactionRelationshipType,
    ) {
        // Remove existing
        self.relationships
            .retain(|r| !(r.involves(faction_a) && r.involves(faction_b)));

        let rel = FactionRelationship::new(faction_a, faction_b, relationship);
        self.relationships.push(rel);
    }

    /// Get the relationship between two factions.
    pub fn get_relationship(
        &self,
        faction_a: u32,
        faction_b: u32,
    ) -> FactionRelationshipType {
        self.relationships
            .iter()
            .find(|r| r.involves(faction_a) && r.involves(faction_b))
            .map(|r| r.relationship)
            .unwrap_or(FactionRelationshipType::Neutral)
    }

    /// Modify the relationship between two factions.
    pub fn modify_relationship(
        &mut self,
        faction_a: u32,
        faction_b: u32,
        delta: f32,
        reason: impl Into<String>,
    ) {
        if let Some(rel) = self
            .relationships
            .iter_mut()
            .find(|r| r.involves(faction_a) && r.involves(faction_b))
        {
            rel.modify(delta, reason);
        } else {
            let mut rel =
                FactionRelationship::new(faction_a, faction_b, FactionRelationshipType::Neutral);
            rel.modify(delta, reason);
            self.relationships.push(rel);
        }
    }

    /// Check if two factions are hostile.
    pub fn are_hostile(&self, faction_a: u32, faction_b: u32) -> bool {
        self.get_relationship(faction_a, faction_b).is_hostile()
    }

    /// Get allies of a faction.
    pub fn allies_of(&self, faction_id: u32) -> Vec<u32> {
        self.relationships
            .iter()
            .filter(|r| {
                r.involves(faction_id)
                    && matches!(
                        r.relationship,
                        FactionRelationshipType::Ally | FactionRelationshipType::Friendly
                    )
            })
            .filter_map(|r| r.other_faction(faction_id))
            .collect()
    }

    /// Get enemies of a faction.
    pub fn enemies_of(&self, faction_id: u32) -> Vec<u32> {
        self.relationships
            .iter()
            .filter(|r| r.involves(faction_id) && r.relationship.is_hostile())
            .filter_map(|r| r.other_faction(faction_id))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Entity faction membership
    // -----------------------------------------------------------------------

    /// Get which faction(s) an entity belongs to.
    pub fn entity_factions(&self, entity_id: u64) -> Vec<u32> {
        self.factions
            .iter()
            .filter(|(_, f)| f.is_member(entity_id))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Check if two entities are in hostile factions.
    pub fn entities_hostile(&self, a: u64, b: u64) -> bool {
        let factions_a = self.entity_factions(a);
        let factions_b = self.entity_factions(b);

        for fa in &factions_a {
            for fb in &factions_b {
                if fa != fb && self.are_hostile(*fa, *fb) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if two entities are in allied factions.
    pub fn entities_allied(&self, a: u64, b: u64) -> bool {
        let factions_a = self.entity_factions(a);
        let factions_b = self.entity_factions(b);

        for fa in &factions_a {
            for fb in &factions_b {
                if fa == fb {
                    return true; // same faction
                }
                if matches!(
                    self.get_relationship(*fa, *fb),
                    FactionRelationshipType::Ally
                ) {
                    return true;
                }
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /// Process pending events and update state.
    pub fn update(&mut self, _dt: f32, _game_time: f64) {
        // Process pending events
        while let Some(mut event) = self.pending_events.pop_front() {
            event.processed = true;
            self.event_history.push_back(event);
        }

        // Cap history
        while self.event_history.len() > MAX_REPUTATION_HISTORY {
            self.event_history.pop_front();
        }
    }

    /// Drain pending events for external processing.
    pub fn drain_events(&mut self) -> Vec<ReputationEvent> {
        self.pending_events.drain(..).collect()
    }

    /// Get recent reputation events for an entity.
    pub fn recent_events(&self, entity_id: u64, count: usize) -> Vec<&ReputationEvent> {
        self.event_history
            .iter()
            .filter(|e| e.entity_id == entity_id)
            .rev()
            .take(count)
            .collect()
    }

    /// Find which faction controls a world position.
    pub fn faction_at(&self, x: f32, z: f32) -> Option<u32> {
        self.factions
            .iter()
            .find(|(_, f)| f.is_in_territory(x, z))
            .map(|(id, _)| *id)
    }
}

impl Default for FactionSystem {
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
    fn test_reputation_standing() {
        assert_eq!(
            ReputationStanding::from_value(0.0),
            ReputationStanding::Neutral
        );
        assert_eq!(
            ReputationStanding::from_value(70.0),
            ReputationStanding::Allied
        );
        assert_eq!(
            ReputationStanding::from_value(-70.0),
            ReputationStanding::Hostile
        );
    }

    #[test]
    fn test_faction_membership() {
        let mut system = FactionSystem::new();
        let id = system.register_faction("Test Guild");

        let faction = system.get_faction_mut(id).unwrap();
        faction.add_member(1);
        faction.add_member(2);

        assert!(faction.is_member(1));
        assert_eq!(faction.member_count(), 2);

        faction.promote(1);
        assert_eq!(faction.get_rank(1), Some(1));
    }

    #[test]
    fn test_reputation_modification() {
        let mut system = FactionSystem::new();
        let faction_id = system.register_faction("Guards");

        system.apply_reputation_event(1, faction_id, ReputationReason::KilledMember, 0.0);

        let rep = system.get_reputation(1, faction_id);
        assert!(rep < 0.0);

        let standing = system.get_standing(1, faction_id);
        assert_eq!(standing, ReputationStanding::Unfriendly);
    }

    #[test]
    fn test_faction_relationships() {
        let mut system = FactionSystem::new();
        let guards = system.register_faction("Guards");
        let thieves = system.register_faction("Thieves");

        system.set_relationship(guards, thieves, FactionRelationshipType::Enemy);
        assert!(system.are_hostile(guards, thieves));

        let enemies = system.enemies_of(guards);
        assert!(enemies.contains(&thieves));
    }

    #[test]
    fn test_territory() {
        let mut system = FactionSystem::new();
        let id = system.register_faction("Kingdom");

        let faction = system.get_faction_mut(id).unwrap();
        faction.add_territory(FactionTerritory::new(1, "Castle", (50.0, 50.0), 30.0));

        assert!(faction.is_in_territory(60.0, 60.0));
        assert!(!faction.is_in_territory(200.0, 200.0));
    }

    #[test]
    fn test_reputation_splash() {
        let mut system = FactionSystem::new();
        let guards = system.register_faction("Guards");
        let militia = system.register_faction("Militia");

        system.set_relationship(guards, militia, FactionRelationshipType::Ally);

        // Killing a guard should also hurt militia rep
        system.apply_reputation_event(1, guards, ReputationReason::KilledMember, 0.0);

        let militia_rep = system.get_reputation(1, militia);
        assert!(militia_rep < 0.0);
    }
}
