// engine/gameplay/src/collectibles.rs
//
// Collectible system for the Genovo engine.
//
// Provides tracking and management of collectible items in the game:
//
// - Collectible type definitions with categories.
// - Per-player collection tracking.
// - Completion percentage calculation.
// - Rewards triggered on completing a set.
// - Map marker data for undiscovered collectibles.
// - Persistence-friendly serializable state.
// - Collection events for UI integration.
// - Region-based collectible grouping.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum collectible types in the game.
const MAX_COLLECTIBLE_TYPES: usize = 2048;

/// Maximum collectibles in a single set.
const MAX_SET_SIZE: usize = 256;

/// Default radar detection radius for nearby collectibles.
const DEFAULT_RADAR_RADIUS: f32 = 50.0;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a collectible definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollectibleId(pub u32);

/// Unique identifier for a collectible set (group).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollectibleSetId(pub u32);

/// Unique identifier for a collectible category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollectibleCategoryId(pub u32);

// ---------------------------------------------------------------------------
// Collectible Category
// ---------------------------------------------------------------------------

/// A category grouping related collectibles.
#[derive(Debug, Clone)]
pub struct CollectibleCategory {
    /// Category identifier.
    pub id: CollectibleCategoryId,
    /// Display name.
    pub name: String,
    /// Localization key.
    pub loc_key: Option<String>,
    /// Description.
    pub description: String,
    /// Icon identifier.
    pub icon: String,
    /// Sort order in the collection UI.
    pub sort_order: i32,
    /// Whether this category is visible in the UI.
    pub visible: bool,
}

// ---------------------------------------------------------------------------
// Collectible Rarity
// ---------------------------------------------------------------------------

/// Rarity tier for collectibles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollectibleRarity {
    /// Common collectible.
    Common,
    /// Uncommon collectible.
    Uncommon,
    /// Rare collectible.
    Rare,
    /// Epic collectible.
    Epic,
    /// Legendary collectible.
    Legendary,
    /// Mythic (ultra-rare) collectible.
    Mythic,
}

impl CollectibleRarity {
    /// Base XP reward multiplier for this rarity.
    pub fn xp_multiplier(self) -> f32 {
        match self {
            Self::Common => 1.0,
            Self::Uncommon => 1.5,
            Self::Rare => 2.5,
            Self::Epic => 5.0,
            Self::Legendary => 10.0,
            Self::Mythic => 25.0,
        }
    }

    /// Color associated with this rarity.
    pub fn color(self) -> [f32; 4] {
        match self {
            Self::Common => [0.8, 0.8, 0.8, 1.0],
            Self::Uncommon => [0.2, 0.8, 0.2, 1.0],
            Self::Rare => [0.2, 0.4, 1.0, 1.0],
            Self::Epic => [0.6, 0.2, 0.8, 1.0],
            Self::Legendary => [1.0, 0.6, 0.0, 1.0],
            Self::Mythic => [1.0, 0.2, 0.2, 1.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Map Marker
// ---------------------------------------------------------------------------

/// Map marker data for a collectible.
#[derive(Debug, Clone)]
pub struct CollectibleMapMarker {
    /// World position.
    pub position: [f32; 3],
    /// Region/zone name.
    pub region: String,
    /// Whether the marker is revealed on the map.
    pub revealed: bool,
    /// Whether the marker shows as collected.
    pub collected: bool,
    /// Custom icon override (empty = use default).
    pub icon_override: String,
    /// Hint text (shown when hovering the marker).
    pub hint: String,
    /// Hint localization key.
    pub hint_loc_key: Option<String>,
    /// Distance at which the marker appears on the minimap.
    pub minimap_radius: f32,
}

impl Default for CollectibleMapMarker {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            region: String::new(),
            revealed: false,
            collected: false,
            icon_override: String::new(),
            hint: String::new(),
            hint_loc_key: None,
            minimap_radius: DEFAULT_RADAR_RADIUS,
        }
    }
}

// ---------------------------------------------------------------------------
// Reward
// ---------------------------------------------------------------------------

/// Reward granted for completing a collectible set.
#[derive(Debug, Clone)]
pub enum CollectibleReward {
    /// Grant XP.
    Experience(u64),
    /// Grant an item.
    Item { item_id: String, count: u32 },
    /// Grant currency.
    Currency { currency_type: String, amount: u64 },
    /// Unlock an achievement.
    Achievement(String),
    /// Unlock a cosmetic.
    Cosmetic(String),
    /// Grant a title.
    Title(String),
    /// Unlock a recipe.
    Recipe(String),
    /// Grant a skill point.
    SkillPoint(u32),
    /// Custom reward.
    Custom { name: String, params: HashMap<String, String> },
}

// ---------------------------------------------------------------------------
// Collectible Definition
// ---------------------------------------------------------------------------

/// Definition of a single collectible.
#[derive(Debug, Clone)]
pub struct CollectibleDefinition {
    /// Unique identifier.
    pub id: CollectibleId,
    /// Display name.
    pub name: String,
    /// Localization key for name.
    pub name_loc_key: Option<String>,
    /// Description / lore text.
    pub description: String,
    /// Description localization key.
    pub desc_loc_key: Option<String>,
    /// Category this collectible belongs to.
    pub category: CollectibleCategoryId,
    /// Set this collectible belongs to (if any).
    pub set_id: Option<CollectibleSetId>,
    /// Rarity tier.
    pub rarity: CollectibleRarity,
    /// Map marker data.
    pub marker: CollectibleMapMarker,
    /// XP reward for collecting this item.
    pub xp_reward: u64,
    /// Additional rewards on collection.
    pub rewards: Vec<CollectibleReward>,
    /// Icon identifier.
    pub icon: String,
    /// 3D model identifier (for world display).
    pub model: String,
    /// Whether this collectible is a secret (hidden from lists until found).
    pub is_secret: bool,
    /// Required prerequisites to make this collectible spawnable.
    pub prerequisites: Vec<String>,
    /// Sort order within its category.
    pub sort_order: i32,
}

// ---------------------------------------------------------------------------
// Collectible Set
// ---------------------------------------------------------------------------

/// A set of related collectibles with completion rewards.
#[derive(Debug, Clone)]
pub struct CollectibleSet {
    /// Set identifier.
    pub id: CollectibleSetId,
    /// Set name.
    pub name: String,
    /// Localization key.
    pub name_loc_key: Option<String>,
    /// Description.
    pub description: String,
    /// Category.
    pub category: CollectibleCategoryId,
    /// Collectible IDs in this set.
    pub members: Vec<CollectibleId>,
    /// Rewards for completing the set (collecting all members).
    pub completion_rewards: Vec<CollectibleReward>,
    /// Milestone rewards (given at partial completion).
    pub milestone_rewards: Vec<MilestoneReward>,
    /// Icon.
    pub icon: String,
    /// Whether to show progress in the UI.
    pub show_progress: bool,
}

/// A milestone reward given at partial set completion.
#[derive(Debug, Clone)]
pub struct MilestoneReward {
    /// Number of collectibles required for this milestone.
    pub required_count: u32,
    /// Rewards granted.
    pub rewards: Vec<CollectibleReward>,
    /// Description text.
    pub description: String,
}

impl CollectibleSet {
    /// Get the total size of the set.
    pub fn total_count(&self) -> usize {
        self.members.len()
    }
}

// ---------------------------------------------------------------------------
// Collection State
// ---------------------------------------------------------------------------

/// Tracks which collectibles have been found by a player.
#[derive(Debug, Clone)]
pub struct CollectionState {
    /// Collected items (set of collected IDs).
    pub collected: Vec<CollectibleId>,
    /// Revealed but not yet collected.
    pub revealed: Vec<CollectibleId>,
    /// Completed sets.
    pub completed_sets: Vec<CollectibleSetId>,
    /// Claimed milestones.
    pub claimed_milestones: HashMap<CollectibleSetId, Vec<u32>>,
    /// Total XP earned from collectibles.
    pub total_xp_earned: u64,
    /// Discovery timestamps (collectible ID -> real time in seconds).
    pub discovery_times: HashMap<CollectibleId, f64>,
}

impl CollectionState {
    /// Create a new empty collection state.
    pub fn new() -> Self {
        Self {
            collected: Vec::new(),
            revealed: Vec::new(),
            completed_sets: Vec::new(),
            claimed_milestones: HashMap::new(),
            total_xp_earned: 0,
            discovery_times: HashMap::new(),
        }
    }

    /// Check if a collectible has been collected.
    pub fn is_collected(&self, id: CollectibleId) -> bool {
        self.collected.contains(&id)
    }

    /// Check if a collectible has been revealed.
    pub fn is_revealed(&self, id: CollectibleId) -> bool {
        self.revealed.contains(&id) || self.collected.contains(&id)
    }

    /// Check if a set is completed.
    pub fn is_set_complete(&self, set_id: CollectibleSetId) -> bool {
        self.completed_sets.contains(&set_id)
    }

    /// Collect a collectible. Returns true if it was newly collected.
    pub fn collect(&mut self, id: CollectibleId, time: f64) -> bool {
        if self.is_collected(id) {
            return false;
        }
        self.collected.push(id);
        self.discovery_times.insert(id, time);
        self.revealed.retain(|&r| r != id);
        true
    }

    /// Reveal a collectible on the map.
    pub fn reveal(&mut self, id: CollectibleId) {
        if !self.is_revealed(id) {
            self.revealed.push(id);
        }
    }

    /// Get the count of collected items in a set.
    pub fn set_progress(&self, set: &CollectibleSet) -> (u32, u32) {
        let collected_count = set.members.iter().filter(|&&m| self.is_collected(m)).count() as u32;
        (collected_count, set.total_count() as u32)
    }

    /// Completion percentage for a set (0.0 to 100.0).
    pub fn set_completion_percent(&self, set: &CollectibleSet) -> f32 {
        let (collected, total) = self.set_progress(set);
        if total == 0 { return 0.0; }
        collected as f32 / total as f32 * 100.0
    }

    /// Total collected count.
    pub fn total_collected(&self) -> usize {
        self.collected.len()
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Events emitted by the collectible system.
#[derive(Debug, Clone)]
pub enum CollectibleEvent {
    /// A collectible was collected.
    Collected { id: CollectibleId, name: String, rarity: CollectibleRarity },
    /// A collectible was revealed on the map.
    Revealed { id: CollectibleId },
    /// A set was completed.
    SetCompleted { set_id: CollectibleSetId, name: String },
    /// A milestone was reached.
    MilestoneReached { set_id: CollectibleSetId, count: u32 },
    /// A reward was granted.
    RewardGranted { reward: CollectibleReward },
    /// Completion percentage updated.
    ProgressUpdated { category: CollectibleCategoryId, percent: f32 },
}

// ---------------------------------------------------------------------------
// Collectible System
// ---------------------------------------------------------------------------

/// Main collectible management system.
#[derive(Debug)]
pub struct CollectibleSystem {
    /// Collectible definitions.
    pub definitions: HashMap<CollectibleId, CollectibleDefinition>,
    /// Collectible sets.
    pub sets: HashMap<CollectibleSetId, CollectibleSet>,
    /// Categories.
    pub categories: HashMap<CollectibleCategoryId, CollectibleCategory>,
    /// Player collection state.
    pub state: CollectionState,
    /// Pending events.
    pub events: Vec<CollectibleEvent>,
    /// Current game time.
    pub game_time: f64,
}

impl CollectibleSystem {
    /// Create a new collectible system.
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            sets: HashMap::new(),
            categories: HashMap::new(),
            state: CollectionState::new(),
            events: Vec::new(),
            game_time: 0.0,
        }
    }

    /// Register a category.
    pub fn register_category(&mut self, category: CollectibleCategory) {
        self.categories.insert(category.id, category);
    }

    /// Register a collectible definition.
    pub fn register_collectible(&mut self, definition: CollectibleDefinition) {
        self.definitions.insert(definition.id, definition);
    }

    /// Register a collectible set.
    pub fn register_set(&mut self, set: CollectibleSet) {
        self.sets.insert(set.id, set);
    }

    /// Collect a collectible by ID.
    pub fn collect(&mut self, id: CollectibleId) {
        if self.state.is_collected(id) {
            return;
        }

        let def = match self.definitions.get(&id) {
            Some(d) => d.clone(),
            None => return,
        };

        self.state.collect(id, self.game_time);
        self.state.total_xp_earned += def.xp_reward;

        self.events.push(CollectibleEvent::Collected {
            id,
            name: def.name.clone(),
            rarity: def.rarity,
        });

        for reward in &def.rewards {
            self.events.push(CollectibleEvent::RewardGranted { reward: reward.clone() });
        }

        // Check set completion.
        if let Some(set_id) = def.set_id {
            if let Some(set) = self.sets.get(&set_id).cloned() {
                let (collected, total) = self.state.set_progress(&set);

                // Check milestones.
                for milestone in &set.milestone_rewards {
                    if collected >= milestone.required_count {
                        let claimed = self.state.claimed_milestones
                            .entry(set_id)
                            .or_insert_with(Vec::new);
                        if !claimed.contains(&milestone.required_count) {
                            claimed.push(milestone.required_count);
                            self.events.push(CollectibleEvent::MilestoneReached {
                                set_id,
                                count: milestone.required_count,
                            });
                            for reward in &milestone.rewards {
                                self.events.push(CollectibleEvent::RewardGranted { reward: reward.clone() });
                            }
                        }
                    }
                }

                // Check full completion.
                if collected == total && !self.state.is_set_complete(set_id) {
                    self.state.completed_sets.push(set_id);
                    self.events.push(CollectibleEvent::SetCompleted {
                        set_id,
                        name: set.name.clone(),
                    });
                    for reward in &set.completion_rewards {
                        self.events.push(CollectibleEvent::RewardGranted { reward: reward.clone() });
                    }
                }
            }
        }
    }

    /// Reveal a collectible on the map.
    pub fn reveal(&mut self, id: CollectibleId) {
        if !self.state.is_revealed(id) {
            self.state.reveal(id);
            self.events.push(CollectibleEvent::Revealed { id });
        }
    }

    /// Get overall completion percentage across all collectibles.
    pub fn overall_completion(&self) -> f32 {
        let total = self.definitions.len();
        if total == 0 { return 0.0; }
        self.state.total_collected() as f32 / total as f32 * 100.0
    }

    /// Get completion percentage for a specific category.
    pub fn category_completion(&self, category_id: CollectibleCategoryId) -> f32 {
        let total: usize = self.definitions.values()
            .filter(|d| d.category == category_id)
            .count();
        if total == 0 { return 0.0; }
        let collected: usize = self.definitions.values()
            .filter(|d| d.category == category_id && self.state.is_collected(d.id))
            .count();
        collected as f32 / total as f32 * 100.0
    }

    /// Find nearby uncollected collectibles.
    pub fn nearby_uncollected(&self, position: [f32; 3], radius: f32) -> Vec<CollectibleId> {
        self.definitions.values()
            .filter(|d| {
                if self.state.is_collected(d.id) { return false; }
                let dx = d.marker.position[0] - position[0];
                let dy = d.marker.position[1] - position[1];
                let dz = d.marker.position[2] - position[2];
                (dx * dx + dy * dy + dz * dz) <= radius * radius
            })
            .map(|d| d.id)
            .collect()
    }

    /// Update the system.
    pub fn update(&mut self, dt: f32) {
        self.game_time += dt as f64;
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<CollectibleEvent> {
        std::mem::take(&mut self.events)
    }
}
