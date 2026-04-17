//! Cooking and alchemy system for combining ingredients into consumable items.
//!
//! Provides:
//! - **Ingredient properties**: spicy, sweet, healing, toxic, etc. with magnitudes
//! - **Property combination rules**: how properties interact when combined
//! - **Ingredient discovery**: unknown ingredients reveal properties when used
//! - **Recipe rating**: 1-5 star quality based on ingredient quality and synergy
//! - **Food buffs/debuffs**: temporary effects from consuming cooked items
//! - **Cooking stations**: different station types with fuel and heat mechanics
//! - **Recipe book**: unlockable recipes with mastery progression
//! - **Experimentation**: discover new recipes by combining unknown ingredients
//! - **Spoilage system**: food expires over time based on freshness
//! - **Chef skill progression**: better results at higher skill levels
//! - **Batch cooking**: cook multiple servings at once
//! - **Cooking mini-game data**: timing windows, temperature control
//! - **Ingredient quality grades**: Poor/Normal/Fine/Superior/Legendary
//!
//! # Design
//!
//! The core of the system is the [`CookingEngine`] which takes a set of
//! [`Ingredient`]s and a [`CookingStation`], evaluates property combinations
//! using the [`PropertyCombinationRules`], and produces a [`CookedDish`]
//! with a quality rating and associated buffs/debuffs.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum ingredients per recipe.
pub const MAX_INGREDIENTS: usize = 8;
/// Maximum properties per ingredient.
pub const MAX_PROPERTIES: usize = 6;
/// Minimum recipe rating.
pub const MIN_RATING: u8 = 1;
/// Maximum recipe rating (5 stars).
pub const MAX_RATING: u8 = 5;
/// Default fuel consumption rate per cook.
pub const DEFAULT_FUEL_COST: f32 = 1.0;
/// Maximum buff/debuff duration in seconds.
pub const MAX_EFFECT_DURATION: f32 = 600.0;
/// Minimum effect duration.
pub const MIN_EFFECT_DURATION: f32 = 10.0;
/// Experience per successful cook.
pub const XP_PER_COOK: u32 = 10;
/// Bonus XP for a 5-star dish.
pub const XP_BONUS_FIVE_STAR: u32 = 50;
/// Number of mastery levels per recipe.
pub const MAX_MASTERY_LEVEL: u32 = 5;
/// Cooks needed per mastery level.
pub const COOKS_PER_MASTERY: u32 = 10;
/// Maximum active buffs from food.
pub const MAX_ACTIVE_FOOD_BUFFS: usize = 3;
/// Small epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;
/// Base spoilage time in seconds (24 hours game time).
pub const BASE_SPOILAGE_TIME: f32 = 86400.0;
/// Maximum batch size.
pub const MAX_BATCH_SIZE: u32 = 10;
/// Maximum chef skill level.
pub const MAX_CHEF_LEVEL: u32 = 50;
/// Bonus quality per chef level.
pub const QUALITY_PER_CHEF_LEVEL: f32 = 0.02;
/// Mini-game perfect window (seconds).
pub const PERFECT_TIMING_WINDOW: f32 = 0.5;
/// Mini-game good window.
pub const GOOD_TIMING_WINDOW: f32 = 1.0;

// ---------------------------------------------------------------------------
// IngredientProperty
// ---------------------------------------------------------------------------

/// A property that an ingredient can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IngredientPropertyType {
    Healing,
    Toxic,
    Spicy,
    Sweet,
    Sour,
    Bitter,
    Salty,
    Umami,
    AttackBoost,
    DefenseBoost,
    SpeedBoost,
    StaminaRestore,
    ManaRestore,
    FireResist,
    ColdResist,
    ElectricResist,
    Stealth,
    Strength,
    Luck,
    Energizing,
    Calming,
}

impl IngredientPropertyType {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Healing => "Healing",
            Self::Toxic => "Toxic",
            Self::Spicy => "Spicy",
            Self::Sweet => "Sweet",
            Self::Sour => "Sour",
            Self::Bitter => "Bitter",
            Self::Salty => "Salty",
            Self::Umami => "Umami",
            Self::AttackBoost => "Attack Boost",
            Self::DefenseBoost => "Defense Boost",
            Self::SpeedBoost => "Speed Boost",
            Self::StaminaRestore => "Stamina Restore",
            Self::ManaRestore => "Mana Restore",
            Self::FireResist => "Fire Resistance",
            Self::ColdResist => "Cold Resistance",
            Self::ElectricResist => "Electric Resistance",
            Self::Stealth => "Stealth",
            Self::Strength => "Strength",
            Self::Luck => "Luck",
            Self::Energizing => "Energizing",
            Self::Calming => "Calming",
        }
    }

    pub fn is_flavor(&self) -> bool {
        matches!(
            self,
            Self::Spicy | Self::Sweet | Self::Sour | Self::Bitter | Self::Salty | Self::Umami
        )
    }

    pub fn is_effect(&self) -> bool {
        !self.is_flavor() && *self != Self::Toxic
    }

    /// Get all flavor types.
    pub fn all_flavors() -> &'static [IngredientPropertyType] {
        &[
            Self::Spicy, Self::Sweet, Self::Sour,
            Self::Bitter, Self::Salty, Self::Umami,
        ]
    }
}

/// An ingredient property with a magnitude.
#[derive(Debug, Clone)]
pub struct IngredientProperty {
    pub property_type: IngredientPropertyType,
    pub magnitude: f32,
    pub discovered: bool,
}

impl IngredientProperty {
    pub fn new(property_type: IngredientPropertyType, magnitude: f32) -> Self {
        Self {
            property_type,
            magnitude: magnitude.clamp(0.0, 1.0),
            discovered: false,
        }
    }

    pub fn discovered(property_type: IngredientPropertyType, magnitude: f32) -> Self {
        Self {
            property_type,
            magnitude: magnitude.clamp(0.0, 1.0),
            discovered: true,
        }
    }
}

// ---------------------------------------------------------------------------
// IngredientQuality
// ---------------------------------------------------------------------------

/// Quality tier of an ingredient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IngredientQuality {
    Poor,
    Normal,
    Fine,
    Superior,
    Legendary,
}

impl IngredientQuality {
    pub fn multiplier(&self) -> f32 {
        match self {
            Self::Poor => 0.5,
            Self::Normal => 1.0,
            Self::Fine => 1.3,
            Self::Superior => 1.6,
            Self::Legendary => 2.0,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Poor => "Poor",
            Self::Normal => "Normal",
            Self::Fine => "Fine",
            Self::Superior => "Superior",
            Self::Legendary => "Legendary",
        }
    }

    /// Spoilage rate multiplier (poor spoils faster, legendary slower).
    pub fn spoilage_multiplier(&self) -> f32 {
        match self {
            Self::Poor => 2.0,
            Self::Normal => 1.0,
            Self::Fine => 0.8,
            Self::Superior => 0.5,
            Self::Legendary => 0.2,
        }
    }

    /// Upgrade quality tier (if possible).
    pub fn upgrade(&self) -> Self {
        match self {
            Self::Poor => Self::Normal,
            Self::Normal => Self::Fine,
            Self::Fine => Self::Superior,
            Self::Superior => Self::Legendary,
            Self::Legendary => Self::Legendary,
        }
    }

    /// Downgrade quality tier.
    pub fn downgrade(&self) -> Self {
        match self {
            Self::Poor => Self::Poor,
            Self::Normal => Self::Poor,
            Self::Fine => Self::Normal,
            Self::Superior => Self::Fine,
            Self::Legendary => Self::Superior,
        }
    }
}

// ---------------------------------------------------------------------------
// Spoilage System
// ---------------------------------------------------------------------------

/// Tracks spoilage state of a food item.
#[derive(Debug, Clone)]
pub struct SpoilageState {
    /// Time when the item was created (game time).
    pub created_at: f64,
    /// Total shelf life in seconds.
    pub shelf_life: f32,
    /// Whether the item has been preserved (doubles shelf life).
    pub preserved: bool,
    /// Current freshness (1.0 = fresh, 0.0 = spoiled).
    pub freshness: f32,
    /// Whether the item is refrigerated (slows spoilage).
    pub refrigerated: bool,
}

impl SpoilageState {
    pub fn new(created_at: f64, base_shelf_life: f32, quality: IngredientQuality) -> Self {
        let shelf_life = base_shelf_life / quality.spoilage_multiplier();
        Self {
            created_at,
            shelf_life,
            preserved: false,
            freshness: 1.0,
            refrigerated: false,
        }
    }

    /// Update freshness based on current game time.
    pub fn update(&mut self, current_time: f64) {
        let elapsed = (current_time - self.created_at) as f32;
        let effective_life = if self.preserved {
            self.shelf_life * 2.0
        } else if self.refrigerated {
            self.shelf_life * 1.5
        } else {
            self.shelf_life
        };

        self.freshness = (1.0 - elapsed / effective_life).clamp(0.0, 1.0);
    }

    /// Whether the item is spoiled.
    pub fn is_spoiled(&self) -> bool {
        self.freshness <= 0.0
    }

    /// Whether the item is stale (below 50% freshness).
    pub fn is_stale(&self) -> bool {
        self.freshness < 0.5
    }

    /// Preserve the item (salt, smoke, etc.).
    pub fn preserve(&mut self) {
        self.preserved = true;
    }

    /// The quality penalty from spoilage (0 = no penalty, 1 = full penalty).
    pub fn quality_penalty(&self) -> f32 {
        (1.0 - self.freshness).max(0.0)
    }

    /// Estimated hours remaining before spoilage.
    pub fn hours_remaining(&self, current_time: f64) -> f32 {
        let elapsed = (current_time - self.created_at) as f32;
        let effective_life = if self.preserved {
            self.shelf_life * 2.0
        } else if self.refrigerated {
            self.shelf_life * 1.5
        } else {
            self.shelf_life
        };
        ((effective_life - elapsed) / 3600.0).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Ingredient
// ---------------------------------------------------------------------------

/// An ingredient that can be used in cooking.
#[derive(Debug, Clone)]
pub struct CookingIngredient {
    pub item_id: String,
    pub name: String,
    pub description: String,
    pub category: IngredientCategory,
    pub properties: Vec<IngredientProperty>,
    pub quality: IngredientQuality,
    pub base_value: u32,
    pub quantity: u32,
    pub fully_analyzed: bool,
    pub icon: String,
    pub weight: f32,
    /// Spoilage state (None = non-perishable).
    pub spoilage: Option<SpoilageState>,
}

/// Category of ingredient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IngredientCategory {
    Vegetable,
    Fruit,
    Meat,
    Fish,
    Dairy,
    Grain,
    Spice,
    Herb,
    Mushroom,
    Mineral,
    Insect,
    Monster,
    Magical,
    Liquid,
}

impl IngredientCategory {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Vegetable => "Vegetable",
            Self::Fruit => "Fruit",
            Self::Meat => "Meat",
            Self::Fish => "Fish",
            Self::Dairy => "Dairy",
            Self::Grain => "Grain",
            Self::Spice => "Spice",
            Self::Herb => "Herb",
            Self::Mushroom => "Mushroom",
            Self::Mineral => "Mineral",
            Self::Insect => "Insect",
            Self::Monster => "Monster Part",
            Self::Magical => "Magical",
            Self::Liquid => "Liquid",
        }
    }

    /// Whether this category is perishable.
    pub fn is_perishable(&self) -> bool {
        matches!(
            self,
            Self::Vegetable | Self::Fruit | Self::Meat | Self::Fish | Self::Dairy | Self::Mushroom
        )
    }

    /// Base shelf life for this category (in seconds of game time).
    pub fn base_shelf_life(&self) -> f32 {
        match self {
            Self::Meat | Self::Fish => BASE_SPOILAGE_TIME * 0.5,
            Self::Dairy => BASE_SPOILAGE_TIME * 0.75,
            Self::Fruit | Self::Vegetable => BASE_SPOILAGE_TIME,
            Self::Mushroom => BASE_SPOILAGE_TIME * 0.8,
            _ => BASE_SPOILAGE_TIME * 10.0, // Non-perishable essentially.
        }
    }

    /// Synergy bonus when combined with another category.
    pub fn category_synergy(&self, other: &IngredientCategory) -> f32 {
        match (self, other) {
            (Self::Meat, Self::Spice) | (Self::Spice, Self::Meat) => 0.2,
            (Self::Fish, Self::Herb) | (Self::Herb, Self::Fish) => 0.15,
            (Self::Vegetable, Self::Grain) | (Self::Grain, Self::Vegetable) => 0.1,
            (Self::Fruit, Self::Dairy) | (Self::Dairy, Self::Fruit) => 0.15,
            (Self::Mushroom, Self::Herb) | (Self::Herb, Self::Mushroom) => 0.2,
            (Self::Monster, Self::Magical) | (Self::Magical, Self::Monster) => 0.25,
            (Self::Liquid, Self::Grain) | (Self::Grain, Self::Liquid) => 0.1,
            _ => 0.0,
        }
    }
}

impl CookingIngredient {
    pub fn new(item_id: impl Into<String>, name: impl Into<String>, category: IngredientCategory) -> Self {
        Self {
            item_id: item_id.into(),
            name: name.into(),
            description: String::new(),
            category,
            properties: Vec::new(),
            quality: IngredientQuality::Normal,
            base_value: 1,
            quantity: 0,
            fully_analyzed: false,
            icon: String::new(),
            weight: 0.1,
            spoilage: None,
        }
    }

    pub fn with_property(mut self, prop: IngredientProperty) -> Self {
        self.properties.push(prop);
        self
    }

    pub fn with_quality(mut self, quality: IngredientQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Initialize spoilage tracking.
    pub fn with_spoilage(mut self, game_time: f64) -> Self {
        if self.category.is_perishable() {
            self.spoilage = Some(SpoilageState::new(
                game_time,
                self.category.base_shelf_life(),
                self.quality,
            ));
        }
        self
    }

    pub fn discovered_properties(&self) -> Vec<&IngredientProperty> {
        self.properties.iter().filter(|p| p.discovered).collect()
    }

    pub fn discover_property(&mut self, property_type: IngredientPropertyType) -> bool {
        for prop in &mut self.properties {
            if prop.property_type == property_type && !prop.discovered {
                prop.discovered = true;
                self.fully_analyzed = self.properties.iter().all(|p| p.discovered);
                return true;
            }
        }
        false
    }

    pub fn discover_all(&mut self) {
        for prop in &mut self.properties {
            prop.discovered = true;
        }
        self.fully_analyzed = true;
    }

    pub fn property_magnitude(&self, prop_type: IngredientPropertyType) -> f32 {
        self.properties
            .iter()
            .find(|p| p.property_type == prop_type)
            .map(|p| p.magnitude)
            .unwrap_or(0.0)
    }

    pub fn quality_multiplier(&self) -> f32 {
        let base = self.quality.multiplier();
        // Apply spoilage penalty.
        let freshness_penalty = self.spoilage.as_ref()
            .map(|s| s.quality_penalty() * 0.5)
            .unwrap_or(0.0);
        (base - freshness_penalty).max(0.1)
    }

    /// Update spoilage state.
    pub fn update_spoilage(&mut self, game_time: f64) {
        if let Some(ref mut spoilage) = self.spoilage {
            spoilage.update(game_time);
            // Downgrade quality if stale.
            if spoilage.is_stale() && self.quality > IngredientQuality::Poor {
                self.quality = self.quality.downgrade();
            }
        }
    }

    /// Whether this ingredient is still usable (not fully spoiled).
    pub fn is_usable(&self) -> bool {
        self.spoilage.as_ref().map(|s| !s.is_spoiled()).unwrap_or(true)
    }
}

// ---------------------------------------------------------------------------
// PropertyCombinationRule
// ---------------------------------------------------------------------------

/// How two ingredient properties interact when combined.
#[derive(Debug, Clone)]
pub struct PropertyCombinationRule {
    pub prop_a: IngredientPropertyType,
    pub prop_b: IngredientPropertyType,
    pub interaction: PropertyInteraction,
}

/// How two properties interact.
#[derive(Debug, Clone)]
pub enum PropertyInteraction {
    Synergy { multiplier: f32 },
    Cancellation { ratio: f32 },
    Transmutation { result: IngredientPropertyType, magnitude: f32 },
    Amplify { target: IngredientPropertyType, multiplier: f32 },
    Conflict { quality_penalty: f32 },
    Neutral,
}

/// Detailed result of combining two properties.
#[derive(Debug, Clone)]
pub struct CombinationResult {
    pub interaction_type: String,
    pub prop_a: IngredientPropertyType,
    pub prop_b: IngredientPropertyType,
    pub magnitude_a_after: f32,
    pub magnitude_b_after: f32,
    pub new_property: Option<(IngredientPropertyType, f32)>,
    pub quality_modifier: f32,
    pub flavor_description: String,
}

/// Database of combination rules.
#[derive(Debug)]
pub struct PropertyCombinationRules {
    rules: HashMap<(IngredientPropertyType, IngredientPropertyType), PropertyInteraction>,
}

impl PropertyCombinationRules {
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    pub fn default_rules() -> Self {
        let mut rules = Self::new();

        // Healing + Toxic cancel out
        rules.add_rule(
            IngredientPropertyType::Healing,
            IngredientPropertyType::Toxic,
            PropertyInteraction::Cancellation { ratio: 0.8 },
        );

        // Spicy + Sweet synergy -> "Sweet heat" (Thai-style flavor)
        rules.add_rule(
            IngredientPropertyType::Spicy,
            IngredientPropertyType::Sweet,
            PropertyInteraction::Synergy { multiplier: 1.3 },
        );

        // Spicy + Sour conflict -> overwhelmingly acidic
        rules.add_rule(
            IngredientPropertyType::Spicy,
            IngredientPropertyType::Sour,
            PropertyInteraction::Conflict { quality_penalty: 0.3 },
        );

        // Sweet + Bitter transmutation -> Energizing
        rules.add_rule(
            IngredientPropertyType::Sweet,
            IngredientPropertyType::Bitter,
            PropertyInteraction::Transmutation {
                result: IngredientPropertyType::Energizing,
                magnitude: 0.5,
            },
        );

        // Healing + Calming synergy -> restorative
        rules.add_rule(
            IngredientPropertyType::Healing,
            IngredientPropertyType::Calming,
            PropertyInteraction::Synergy { multiplier: 1.5 },
        );

        // AttackBoost + SpeedBoost synergy -> berserker
        rules.add_rule(
            IngredientPropertyType::AttackBoost,
            IngredientPropertyType::SpeedBoost,
            PropertyInteraction::Synergy { multiplier: 1.2 },
        );

        // FireResist + ColdResist cancel partially
        rules.add_rule(
            IngredientPropertyType::FireResist,
            IngredientPropertyType::ColdResist,
            PropertyInteraction::Cancellation { ratio: 0.5 },
        );

        // Salty + Umami synergy -> savory depth
        rules.add_rule(
            IngredientPropertyType::Salty,
            IngredientPropertyType::Umami,
            PropertyInteraction::Synergy { multiplier: 1.4 },
        );

        // Toxic + Stealth transmutation -> assassin's veil
        rules.add_rule(
            IngredientPropertyType::Toxic,
            IngredientPropertyType::Stealth,
            PropertyInteraction::Transmutation {
                result: IngredientPropertyType::Stealth,
                magnitude: 0.8,
            },
        );

        // Spicy + Salty synergy -> bold flavor
        rules.add_rule(
            IngredientPropertyType::Spicy,
            IngredientPropertyType::Salty,
            PropertyInteraction::Synergy { multiplier: 1.15 },
        );

        // Sweet + Sour synergy -> tangy (like lemonade)
        rules.add_rule(
            IngredientPropertyType::Sweet,
            IngredientPropertyType::Sour,
            PropertyInteraction::Synergy { multiplier: 1.2 },
        );

        // Bitter + Salty -> acquired taste, slight conflict
        rules.add_rule(
            IngredientPropertyType::Bitter,
            IngredientPropertyType::Salty,
            PropertyInteraction::Conflict { quality_penalty: 0.15 },
        );

        // Energizing + Calming cancel
        rules.add_rule(
            IngredientPropertyType::Energizing,
            IngredientPropertyType::Calming,
            PropertyInteraction::Cancellation { ratio: 0.9 },
        );

        // Healing + ManaRestore synergy -> full restoration
        rules.add_rule(
            IngredientPropertyType::Healing,
            IngredientPropertyType::ManaRestore,
            PropertyInteraction::Synergy { multiplier: 1.3 },
        );

        // AttackBoost + DefenseBoost conflict -> unfocused
        rules.add_rule(
            IngredientPropertyType::AttackBoost,
            IngredientPropertyType::DefenseBoost,
            PropertyInteraction::Conflict { quality_penalty: 0.2 },
        );

        // Strength + StaminaRestore synergy -> endurance
        rules.add_rule(
            IngredientPropertyType::Strength,
            IngredientPropertyType::StaminaRestore,
            PropertyInteraction::Synergy { multiplier: 1.25 },
        );

        // Luck + Stealth transmutation -> fortune's shadow
        rules.add_rule(
            IngredientPropertyType::Luck,
            IngredientPropertyType::Stealth,
            PropertyInteraction::Transmutation {
                result: IngredientPropertyType::Luck,
                magnitude: 0.6,
            },
        );

        // Umami + Sweet synergy -> rich glaze
        rules.add_rule(
            IngredientPropertyType::Umami,
            IngredientPropertyType::Sweet,
            PropertyInteraction::Synergy { multiplier: 1.2 },
        );

        rules
    }

    pub fn add_rule(
        &mut self,
        a: IngredientPropertyType,
        b: IngredientPropertyType,
        interaction: PropertyInteraction,
    ) {
        self.rules.insert((a, b), interaction.clone());
        self.rules.insert((b, a), interaction);
    }

    pub fn get_interaction(
        &self,
        a: IngredientPropertyType,
        b: IngredientPropertyType,
    ) -> &PropertyInteraction {
        self.rules
            .get(&(a, b))
            .unwrap_or(&PropertyInteraction::Neutral)
    }

    /// Evaluate a pair and return a detailed combination result.
    pub fn evaluate_combination(
        &self,
        a: IngredientPropertyType,
        a_magnitude: f32,
        b: IngredientPropertyType,
        b_magnitude: f32,
    ) -> CombinationResult {
        let interaction = self.get_interaction(a, b);
        match interaction {
            PropertyInteraction::Synergy { multiplier } => CombinationResult {
                interaction_type: "Synergy".to_string(),
                prop_a: a,
                prop_b: b,
                magnitude_a_after: a_magnitude * multiplier,
                magnitude_b_after: b_magnitude * multiplier,
                new_property: None,
                quality_modifier: 1.0,
                flavor_description: format!("{} and {} complement each other!", a.display_name(), b.display_name()),
            },
            PropertyInteraction::Cancellation { ratio } => CombinationResult {
                interaction_type: "Cancellation".to_string(),
                prop_a: a,
                prop_b: b,
                magnitude_a_after: a_magnitude * (1.0 - ratio),
                magnitude_b_after: b_magnitude * (1.0 - ratio),
                new_property: None,
                quality_modifier: 1.0,
                flavor_description: format!("{} and {} cancel each other out.", a.display_name(), b.display_name()),
            },
            PropertyInteraction::Transmutation { result, magnitude } => CombinationResult {
                interaction_type: "Transmutation".to_string(),
                prop_a: a,
                prop_b: b,
                magnitude_a_after: a_magnitude * 0.5,
                magnitude_b_after: b_magnitude * 0.5,
                new_property: Some((*result, *magnitude)),
                quality_modifier: 1.1,
                flavor_description: format!("{} and {} transform into {}!", a.display_name(), b.display_name(), result.display_name()),
            },
            PropertyInteraction::Amplify { target, multiplier } => {
                let (ma, mb) = if *target == a {
                    (a_magnitude * multiplier, b_magnitude)
                } else {
                    (a_magnitude, b_magnitude * multiplier)
                };
                CombinationResult {
                    interaction_type: "Amplify".to_string(),
                    prop_a: a,
                    prop_b: b,
                    magnitude_a_after: ma,
                    magnitude_b_after: mb,
                    new_property: None,
                    quality_modifier: 1.0,
                    flavor_description: format!("{} amplifies {}!", b.display_name(), target.display_name()),
                }
            },
            PropertyInteraction::Conflict { quality_penalty } => CombinationResult {
                interaction_type: "Conflict".to_string(),
                prop_a: a,
                prop_b: b,
                magnitude_a_after: a_magnitude,
                magnitude_b_after: b_magnitude,
                new_property: None,
                quality_modifier: 1.0 - quality_penalty,
                flavor_description: format!("{} and {} clash badly!", a.display_name(), b.display_name()),
            },
            PropertyInteraction::Neutral => CombinationResult {
                interaction_type: "Neutral".to_string(),
                prop_a: a,
                prop_b: b,
                magnitude_a_after: a_magnitude,
                magnitude_b_after: b_magnitude,
                new_property: None,
                quality_modifier: 1.0,
                flavor_description: String::new(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Cooking Mini-Game
// ---------------------------------------------------------------------------

/// Data for a cooking mini-game phase.
#[derive(Debug, Clone)]
pub struct CookingMiniGamePhase {
    /// Phase name (e.g., "Searing", "Simmering", "Plating").
    pub name: String,
    /// Target temperature for this phase.
    pub target_temperature: f32,
    /// Acceptable temperature range (+/- this value).
    pub temperature_tolerance: f32,
    /// Duration of this phase in seconds.
    pub duration: f32,
    /// Timing windows for skill checks during this phase.
    pub timing_windows: Vec<TimingWindow>,
    /// Description shown to the player.
    pub instruction: String,
}

/// A timing window during a mini-game phase.
#[derive(Debug, Clone)]
pub struct TimingWindow {
    /// When this window starts (seconds from phase start).
    pub start_time: f32,
    /// Duration of the window.
    pub duration: f32,
    /// Action required (e.g., "flip", "stir", "season").
    pub action: String,
    /// Quality bonus for hitting the perfect window.
    pub perfect_bonus: f32,
    /// Quality bonus for hitting within the good window.
    pub good_bonus: f32,
    /// Quality penalty for missing.
    pub miss_penalty: f32,
}

/// State of an active cooking mini-game.
#[derive(Debug, Clone)]
pub struct CookingMiniGameState {
    /// Current phase index.
    pub current_phase: usize,
    /// Total phases.
    pub total_phases: usize,
    /// Elapsed time in current phase.
    pub phase_elapsed: f32,
    /// Current temperature.
    pub current_temperature: f32,
    /// Accumulated quality modifier from mini-game performance.
    pub quality_modifier: f32,
    /// Whether the mini-game is complete.
    pub complete: bool,
    /// Player inputs received.
    pub inputs_received: Vec<MiniGameInput>,
    /// Number of perfect hits.
    pub perfect_count: u32,
    /// Number of good hits.
    pub good_count: u32,
    /// Number of misses.
    pub miss_count: u32,
}

/// A player input during the mini-game.
#[derive(Debug, Clone)]
pub struct MiniGameInput {
    pub time: f32,
    pub action: String,
    pub result: MiniGameInputResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniGameInputResult {
    Perfect,
    Good,
    Miss,
    WrongAction,
}

impl CookingMiniGameState {
    pub fn new(total_phases: usize) -> Self {
        Self {
            current_phase: 0,
            total_phases,
            phase_elapsed: 0.0,
            current_temperature: 0.0,
            quality_modifier: 1.0,
            complete: false,
            inputs_received: Vec::new(),
            perfect_count: 0,
            good_count: 0,
            miss_count: 0,
        }
    }

    /// Process a player input against the current timing windows.
    pub fn process_input(
        &mut self,
        action: &str,
        phases: &[CookingMiniGamePhase],
    ) -> MiniGameInputResult {
        if self.complete || self.current_phase >= phases.len() {
            return MiniGameInputResult::Miss;
        }

        let phase = &phases[self.current_phase];
        let mut best_result = MiniGameInputResult::Miss;

        for window in &phase.timing_windows {
            if window.action != action {
                continue;
            }

            let window_center = window.start_time + window.duration * 0.5;
            let time_diff = (self.phase_elapsed - window_center).abs();

            if time_diff <= PERFECT_TIMING_WINDOW * 0.5 {
                self.quality_modifier += window.perfect_bonus;
                self.perfect_count += 1;
                best_result = MiniGameInputResult::Perfect;
                break;
            } else if time_diff <= GOOD_TIMING_WINDOW * 0.5 {
                self.quality_modifier += window.good_bonus;
                self.good_count += 1;
                best_result = MiniGameInputResult::Good;
                break;
            }
        }

        if best_result == MiniGameInputResult::Miss {
            // Check if it was the wrong action.
            let has_correct_window = phase.timing_windows.iter().any(|w| {
                let center = w.start_time + w.duration * 0.5;
                let diff = (self.phase_elapsed - center).abs();
                diff <= GOOD_TIMING_WINDOW
            });

            if has_correct_window {
                best_result = MiniGameInputResult::WrongAction;
            }

            self.quality_modifier -= 0.05;
            self.miss_count += 1;
        }

        self.inputs_received.push(MiniGameInput {
            time: self.phase_elapsed,
            action: action.to_string(),
            result: best_result,
        });

        best_result
    }

    /// Update the mini-game state.
    pub fn update(&mut self, dt: f32, phases: &[CookingMiniGamePhase]) {
        if self.complete || self.current_phase >= phases.len() {
            self.complete = true;
            return;
        }

        self.phase_elapsed += dt;

        let phase = &phases[self.current_phase];

        // Check temperature accuracy.
        let temp_diff = (self.current_temperature - phase.target_temperature).abs();
        if temp_diff > phase.temperature_tolerance {
            let penalty = (temp_diff - phase.temperature_tolerance) * 0.001 * dt;
            self.quality_modifier -= penalty;
        }

        // Check for missed timing windows.
        for window in &phase.timing_windows {
            let window_end = window.start_time + window.duration;
            if self.phase_elapsed > window_end + GOOD_TIMING_WINDOW {
                let already_hit = self.inputs_received.iter().any(|i| {
                    i.action == window.action
                        && i.time >= window.start_time
                        && i.time <= window_end + GOOD_TIMING_WINDOW
                });
                if !already_hit {
                    self.quality_modifier -= window.miss_penalty;
                    self.miss_count += 1;
                }
            }
        }

        // Advance to next phase if time is up.
        if self.phase_elapsed >= phase.duration {
            self.current_phase += 1;
            self.phase_elapsed = 0.0;
            if self.current_phase >= self.total_phases {
                self.complete = true;
            }
        }

        self.quality_modifier = self.quality_modifier.clamp(0.1, 2.0);
    }

    /// Get the mini-game performance rating.
    pub fn performance_rating(&self) -> f32 {
        let total = self.perfect_count + self.good_count + self.miss_count;
        if total == 0 {
            return 1.0;
        }
        (self.perfect_count as f32 * 1.0 + self.good_count as f32 * 0.7) / total as f32
    }
}

// ---------------------------------------------------------------------------
// CookingStation
// ---------------------------------------------------------------------------

/// Type of cooking station.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CookingStationType {
    Campfire,
    Stove,
    Oven,
    Cauldron,
    Smoker,
    Grill,
    BrewingStation,
    EnchantingTable,
}

impl CookingStationType {
    pub fn quality_bonus(&self) -> f32 {
        match self {
            Self::Campfire => 0.0,
            Self::Stove => 0.1,
            Self::Oven => 0.15,
            Self::Cauldron => 0.2,
            Self::Smoker => 0.1,
            Self::Grill => 0.1,
            Self::BrewingStation => 0.15,
            Self::EnchantingTable => 0.3,
        }
    }

    pub fn max_complexity(&self) -> usize {
        match self {
            Self::Campfire => 3,
            Self::Stove => 5,
            Self::Oven => 6,
            Self::Cauldron => 8,
            Self::Smoker => 4,
            Self::Grill => 4,
            Self::BrewingStation => 6,
            Self::EnchantingTable => 8,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Campfire => "Campfire",
            Self::Stove => "Stove",
            Self::Oven => "Oven",
            Self::Cauldron => "Cauldron",
            Self::Smoker => "Smoker",
            Self::Grill => "Grill",
            Self::BrewingStation => "Brewing Station",
            Self::EnchantingTable => "Enchanting Table",
        }
    }

    /// Maximum batch size this station supports.
    pub fn max_batch_size(&self) -> u32 {
        match self {
            Self::Campfire => 2,
            Self::Stove => 4,
            Self::Oven => 6,
            Self::Cauldron => 8,
            Self::Smoker => 4,
            Self::Grill => 5,
            Self::BrewingStation => 3,
            Self::EnchantingTable => 2,
        }
    }

    /// Whether this station can preserve (smoke/salt) food.
    pub fn can_preserve(&self) -> bool {
        matches!(self, Self::Smoker | Self::Oven)
    }
}

/// A cooking station instance.
#[derive(Debug, Clone)]
pub struct CookingStation {
    pub station_type: CookingStationType,
    pub fuel: f32,
    pub max_fuel: f32,
    pub fuel_per_cook: f32,
    pub active: bool,
    pub heat: f32,
    pub max_heat: f32,
    pub level: u32,
    pub upgrade_bonus: f32,
    /// Current temperature (for mini-game).
    pub temperature: f32,
    /// Target temperature.
    pub target_temperature: f32,
    /// Heat up rate (degrees per second).
    pub heat_rate: f32,
    /// Cool down rate.
    pub cool_rate: f32,
}

impl CookingStation {
    pub fn new(station_type: CookingStationType) -> Self {
        Self {
            station_type,
            fuel: 10.0,
            max_fuel: 20.0,
            fuel_per_cook: DEFAULT_FUEL_COST,
            active: false,
            heat: 0.0,
            max_heat: 100.0,
            level: 1,
            upgrade_bonus: 0.0,
            temperature: 20.0,
            target_temperature: 180.0,
            heat_rate: 10.0,
            cool_rate: 2.0,
        }
    }

    pub fn add_fuel(&mut self, amount: f32) {
        self.fuel = (self.fuel + amount).min(self.max_fuel);
    }

    pub fn activate(&mut self) -> bool {
        if self.fuel > EPSILON {
            self.active = true;
            self.heat = self.max_heat;
            true
        } else {
            false
        }
    }

    pub fn deactivate(&mut self) {
        self.active = false;
        self.heat = 0.0;
    }

    pub fn consume_fuel(&mut self) -> bool {
        if self.fuel >= self.fuel_per_cook {
            self.fuel -= self.fuel_per_cook;
            if self.fuel <= EPSILON {
                self.deactivate();
            }
            true
        } else {
            false
        }
    }

    /// Consume fuel for a batch cook (larger batches use more fuel).
    pub fn consume_fuel_batch(&mut self, batch_size: u32) -> bool {
        let total_fuel = self.fuel_per_cook * (1.0 + (batch_size as f32 - 1.0) * 0.5);
        if self.fuel >= total_fuel {
            self.fuel -= total_fuel;
            if self.fuel <= EPSILON {
                self.deactivate();
            }
            true
        } else {
            false
        }
    }

    pub fn quality_modifier(&self) -> f32 {
        self.station_type.quality_bonus() + self.upgrade_bonus + (self.level as f32 - 1.0) * 0.05
    }

    pub fn can_handle(&self, ingredient_count: usize) -> bool {
        self.active && ingredient_count <= self.station_type.max_complexity()
    }

    /// Update station temperature simulation.
    pub fn update_temperature(&mut self, dt: f32) {
        if self.active {
            if self.temperature < self.target_temperature {
                self.temperature = (self.temperature + self.heat_rate * dt)
                    .min(self.target_temperature);
            } else if self.temperature > self.target_temperature {
                self.temperature = (self.temperature - self.cool_rate * dt)
                    .max(self.target_temperature);
            }
        } else {
            // Cool down to room temperature.
            if self.temperature > 20.0 {
                self.temperature = (self.temperature - self.cool_rate * dt).max(20.0);
            }
        }
    }

    /// Set target temperature.
    pub fn set_target_temperature(&mut self, temp: f32) {
        self.target_temperature = temp.clamp(20.0, 500.0);
    }
}

// ---------------------------------------------------------------------------
// FoodBuff
// ---------------------------------------------------------------------------

/// A buff or debuff applied by consuming food.
#[derive(Debug, Clone)]
pub struct FoodBuff {
    pub id: String,
    pub name: String,
    pub description: String,
    pub source_property: IngredientPropertyType,
    pub magnitude: f32,
    pub duration: f32,
    pub is_debuff: bool,
    pub icon: String,
    pub remaining_duration: f32,
    pub active: bool,
}

impl FoodBuff {
    pub fn from_property(property: IngredientPropertyType, magnitude: f32, duration: f32) -> Self {
        let is_debuff = matches!(property, IngredientPropertyType::Toxic);
        Self {
            id: format!("food_{:?}_{}", property, (magnitude * 100.0) as u32),
            name: format!(
                "{} {}",
                property.display_name(),
                if is_debuff { "Debuff" } else { "Buff" }
            ),
            description: format!(
                "{} {} by {:.0}% for {:.0}s",
                if is_debuff { "Decreases" } else { "Increases" },
                property.display_name(),
                magnitude * 100.0,
                duration,
            ),
            source_property: property,
            magnitude,
            duration: duration.clamp(MIN_EFFECT_DURATION, MAX_EFFECT_DURATION),
            is_debuff,
            icon: String::new(),
            remaining_duration: 0.0,
            active: false,
        }
    }

    pub fn activate(&mut self) {
        self.active = true;
        self.remaining_duration = self.duration;
    }

    pub fn update(&mut self, dt: f32) {
        if self.active {
            self.remaining_duration -= dt;
            if self.remaining_duration <= 0.0 {
                self.active = false;
                self.remaining_duration = 0.0;
            }
        }
    }

    pub fn remaining_fraction(&self) -> f32 {
        if self.duration > EPSILON {
            self.remaining_duration / self.duration
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// CookedDish
// ---------------------------------------------------------------------------

/// A dish produced by cooking.
#[derive(Debug, Clone)]
pub struct CookedDish {
    pub item_id: String,
    pub name: String,
    pub description: String,
    pub rating: u8,
    pub buffs: Vec<FoodBuff>,
    pub hp_restore: f32,
    pub stamina_restore: f32,
    pub mana_restore: f32,
    pub sell_value: u32,
    pub ingredients_used: Vec<String>,
    pub recipe_id: Option<String>,
    pub is_discovery: bool,
    pub flavor_profile: HashMap<IngredientPropertyType, f32>,
    pub quality_score: f32,
    /// Spoilage state for the cooked dish.
    pub spoilage: SpoilageState,
    /// Number of servings in this batch.
    pub servings: u32,
    /// Combination results for the player's journal.
    pub combination_notes: Vec<String>,
}

impl CookedDish {
    pub fn dominant_flavor(&self) -> Option<IngredientPropertyType> {
        self.flavor_profile
            .iter()
            .filter(|(prop, _)| prop.is_flavor())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(prop, _)| *prop)
    }

    pub fn buff_count(&self) -> usize {
        self.buffs.iter().filter(|b| !b.is_debuff).count()
    }

    pub fn debuff_count(&self) -> usize {
        self.buffs.iter().filter(|b| b.is_debuff).count()
    }

    /// Compute the flavor balance score (how well-balanced the flavors are).
    pub fn flavor_balance(&self) -> f32 {
        let flavor_values: Vec<f32> = self.flavor_profile.iter()
            .filter(|(p, _)| p.is_flavor())
            .map(|(_, &v)| v)
            .collect();

        if flavor_values.len() < 2 {
            return 1.0;
        }

        let max = flavor_values.iter().cloned().fold(0.0f32, f32::max);
        let min = flavor_values.iter().cloned().fold(f32::MAX, f32::min);
        if max < EPSILON {
            return 1.0;
        }

        // Balance is better when the range is small relative to the max.
        1.0 - ((max - min) / max).min(1.0) * 0.5
    }

    /// Consume one serving. Returns true if there are servings remaining.
    pub fn consume_serving(&mut self) -> bool {
        if self.servings > 0 {
            self.servings -= 1;
        }
        self.servings > 0
    }
}

// ---------------------------------------------------------------------------
// CookingRecipe
// ---------------------------------------------------------------------------

/// A known recipe for cooking.
#[derive(Debug, Clone)]
pub struct CookingRecipe {
    pub id: String,
    pub name: String,
    pub description: String,
    pub ingredients: HashMap<String, u32>,
    pub station_type: CookingStationType,
    pub min_level: u32,
    pub base_quality: f32,
    pub unlocked: bool,
    pub times_cooked: u32,
    pub mastery_level: u32,
    pub result_name: String,
    pub result_description: String,
    pub tags: Vec<String>,
    /// Mini-game phases for this recipe (empty = no mini-game).
    pub mini_game_phases: Vec<CookingMiniGamePhase>,
}

impl CookingRecipe {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            ingredients: HashMap::new(),
            station_type: CookingStationType::Campfire,
            min_level: 1,
            base_quality: 0.5,
            unlocked: false,
            times_cooked: 0,
            mastery_level: 0,
            result_name: String::new(),
            result_description: String::new(),
            tags: Vec::new(),
            mini_game_phases: Vec::new(),
        }
    }

    pub fn with_ingredient(mut self, item_id: impl Into<String>, quantity: u32) -> Self {
        self.ingredients.insert(item_id.into(), quantity);
        self
    }

    pub fn with_station(mut self, station: CookingStationType) -> Self {
        self.station_type = station;
        self
    }

    pub fn record_cook(&mut self) {
        self.times_cooked += 1;
        let new_mastery = (self.times_cooked / COOKS_PER_MASTERY).min(MAX_MASTERY_LEVEL);
        if new_mastery > self.mastery_level {
            self.mastery_level = new_mastery;
        }
    }

    pub fn mastery_bonus(&self) -> f32 {
        self.mastery_level as f32 * 0.1
    }

    pub fn can_cook(&self, inventory: &HashMap<String, u32>) -> bool {
        if !self.unlocked {
            return false;
        }
        for (item_id, required) in &self.ingredients {
            let have = inventory.get(item_id).copied().unwrap_or(0);
            if have < *required {
                return false;
            }
        }
        true
    }

    /// Check if batch cooking is possible for a given batch size.
    pub fn can_cook_batch(&self, inventory: &HashMap<String, u32>, batch_size: u32) -> bool {
        if !self.unlocked {
            return false;
        }
        for (item_id, required) in &self.ingredients {
            let have = inventory.get(item_id).copied().unwrap_or(0);
            if have < required * batch_size {
                return false;
            }
        }
        true
    }

    /// Whether this recipe has a mini-game.
    pub fn has_mini_game(&self) -> bool {
        !self.mini_game_phases.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Chef Skill Progression
// ---------------------------------------------------------------------------

/// Tracks the player's chef skill and specializations.
#[derive(Debug, Clone)]
pub struct ChefSkill {
    /// Overall cooking level.
    pub level: u32,
    /// Current XP.
    pub xp: u32,
    /// XP needed for next level.
    pub xp_for_next_level: u32,
    /// Specialization levels.
    pub specializations: HashMap<CookingStationType, u32>,
    /// Total dishes cooked.
    pub total_dishes: u64,
    /// Total 5-star dishes.
    pub five_star_count: u64,
    /// Total discoveries.
    pub discoveries: u64,
    /// Unlocked techniques.
    pub techniques: HashSet<String>,
}

impl ChefSkill {
    pub fn new() -> Self {
        Self {
            level: 1,
            xp: 0,
            xp_for_next_level: 100,
            specializations: HashMap::new(),
            total_dishes: 0,
            five_star_count: 0,
            discoveries: 0,
            techniques: HashSet::new(),
        }
    }

    /// Add XP and handle level ups.
    pub fn add_xp(&mut self, xp: u32) {
        self.xp += xp;
        while self.xp >= self.xp_for_next_level && self.level < MAX_CHEF_LEVEL {
            self.xp -= self.xp_for_next_level;
            self.level += 1;
            self.xp_for_next_level = self.level * 100 + 50;
        }
    }

    /// Get the quality bonus from chef level.
    pub fn quality_bonus(&self) -> f32 {
        self.level as f32 * QUALITY_PER_CHEF_LEVEL
    }

    /// Get the specialization bonus for a station type.
    pub fn specialization_bonus(&self, station: CookingStationType) -> f32 {
        self.specializations.get(&station).copied().unwrap_or(0) as f32 * 0.03
    }

    /// Record cooking a dish.
    pub fn record_cook(&mut self, station: CookingStationType, rating: u8, is_discovery: bool) {
        self.total_dishes += 1;
        if rating == MAX_RATING {
            self.five_star_count += 1;
        }
        if is_discovery {
            self.discoveries += 1;
        }

        // Increase specialization.
        let spec = self.specializations.entry(station).or_insert(0);
        *spec = (*spec + 1).min(20);

        // XP from cooking.
        let mut xp = XP_PER_COOK;
        if rating == MAX_RATING {
            xp += XP_BONUS_FIVE_STAR;
        }
        if is_discovery {
            xp += 25;
        }
        self.add_xp(xp);
    }

    /// Whether the chef has unlocked a technique.
    pub fn has_technique(&self, technique: &str) -> bool {
        self.techniques.contains(technique)
    }

    /// Unlock a technique.
    pub fn unlock_technique(&mut self, technique: impl Into<String>) {
        self.techniques.insert(technique.into());
    }

    /// Chance to upgrade ingredient quality during cooking (higher skill = better chance).
    pub fn quality_upgrade_chance(&self) -> f32 {
        (self.level as f32 * 0.01).min(0.3)
    }

    /// Chance to produce an extra serving (batch efficiency).
    pub fn extra_serving_chance(&self) -> f32 {
        (self.level as f32 * 0.005).min(0.2)
    }
}

impl Default for ChefSkill {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Experimentation System
// ---------------------------------------------------------------------------

/// Tracks the player's experimentation with unknown ingredient combinations.
#[derive(Debug, Clone)]
pub struct ExperimentationLog {
    /// Combinations tried: sorted ingredient IDs -> result.
    tried_combinations: HashMap<String, ExperimentResult>,
    /// Hints about undiscovered recipes.
    hints: Vec<ExperimentHint>,
}

#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub combination_key: String,
    pub dish_name: String,
    pub rating: u8,
    pub times_tried: u32,
    pub best_rating: u8,
    pub timestamp: f64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ExperimentHint {
    pub hint_text: String,
    pub required_ingredients: Vec<String>,
    pub discovered: bool,
}

impl ExperimentationLog {
    pub fn new() -> Self {
        Self {
            tried_combinations: HashMap::new(),
            hints: Vec::new(),
        }
    }

    /// Generate a key from a set of ingredient IDs.
    pub fn combination_key(ingredient_ids: &[String]) -> String {
        let mut sorted: Vec<_> = ingredient_ids.to_vec();
        sorted.sort();
        sorted.join("+")
    }

    /// Record an experiment.
    pub fn record(
        &mut self,
        ingredient_ids: &[String],
        dish_name: &str,
        rating: u8,
        notes: Vec<String>,
        timestamp: f64,
    ) -> bool {
        let key = Self::combination_key(ingredient_ids);
        let is_new = !self.tried_combinations.contains_key(&key);

        let entry = self.tried_combinations.entry(key.clone()).or_insert_with(|| {
            ExperimentResult {
                combination_key: key,
                dish_name: dish_name.to_string(),
                rating,
                times_tried: 0,
                best_rating: rating,
                timestamp,
                notes: Vec::new(),
            }
        });

        entry.times_tried += 1;
        if rating > entry.best_rating {
            entry.best_rating = rating;
        }
        entry.notes.extend(notes);

        // Check hints.
        for hint in &mut self.hints {
            if !hint.discovered {
                let all_present = hint.required_ingredients.iter()
                    .all(|req| ingredient_ids.iter().any(|id| id == req));
                if all_present {
                    hint.discovered = true;
                }
            }
        }

        is_new
    }

    /// Add a hint.
    pub fn add_hint(&mut self, hint: ExperimentHint) {
        self.hints.push(hint);
    }

    /// Get all undiscovered hints.
    pub fn undiscovered_hints(&self) -> Vec<&ExperimentHint> {
        self.hints.iter().filter(|h| !h.discovered).collect()
    }

    /// Total unique combinations tried.
    pub fn total_experiments(&self) -> usize {
        self.tried_combinations.len()
    }

    /// Get the best result for a combination.
    pub fn best_result(&self, ingredient_ids: &[String]) -> Option<&ExperimentResult> {
        let key = Self::combination_key(ingredient_ids);
        self.tried_combinations.get(&key)
    }
}

impl Default for ExperimentationLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CookingEngine
// ---------------------------------------------------------------------------

/// The main cooking engine that produces dishes from ingredients.
pub struct CookingEngine {
    pub rules: PropertyCombinationRules,
    ingredients_db: HashMap<String, CookingIngredient>,
    recipes: HashMap<String, CookingRecipe>,
    chef_skill: ChefSkill,
    active_buffs: Vec<FoodBuff>,
    history: Vec<CookingHistoryEntry>,
    discoveries: HashSet<String>,
    experimentation: ExperimentationLog,
    /// Simple RNG state.
    rng_state: u64,
}

/// Entry in the cooking history.
#[derive(Debug, Clone)]
pub struct CookingHistoryEntry {
    pub dish_name: String,
    pub rating: u8,
    pub was_discovery: bool,
    pub timestamp: f64,
    pub servings: u32,
    pub mini_game_score: Option<f32>,
}

impl CookingEngine {
    pub fn new() -> Self {
        Self {
            rules: PropertyCombinationRules::default_rules(),
            ingredients_db: HashMap::new(),
            recipes: HashMap::new(),
            chef_skill: ChefSkill::new(),
            active_buffs: Vec::new(),
            history: Vec::new(),
            discoveries: HashSet::new(),
            experimentation: ExperimentationLog::new(),
            rng_state: 42,
        }
    }

    fn next_rng(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.rng_state >> 33) as f32) / (u32::MAX as f32)
    }

    pub fn register_ingredient(&mut self, ingredient: CookingIngredient) {
        self.ingredients_db.insert(ingredient.item_id.clone(), ingredient);
    }

    pub fn register_recipe(&mut self, recipe: CookingRecipe) {
        self.recipes.insert(recipe.id.clone(), recipe);
    }

    pub fn unlock_recipe(&mut self, recipe_id: &str) -> bool {
        if let Some(recipe) = self.recipes.get_mut(recipe_id) {
            recipe.unlocked = true;
            true
        } else {
            false
        }
    }

    /// Get the chef skill.
    pub fn chef_skill(&self) -> &ChefSkill {
        &self.chef_skill
    }

    /// Get the experimentation log.
    pub fn experimentation(&self) -> &ExperimentationLog {
        &self.experimentation
    }

    /// Cook a dish using the given ingredients at the given station.
    pub fn cook(
        &mut self,
        ingredient_ids: &[String],
        station: &mut CookingStation,
        game_time: f64,
    ) -> Result<CookedDish, CookingError> {
        self.cook_batch(ingredient_ids, station, game_time, 1, None)
    }

    /// Cook with batch size and optional mini-game result.
    pub fn cook_batch(
        &mut self,
        ingredient_ids: &[String],
        station: &mut CookingStation,
        game_time: f64,
        batch_size: u32,
        mini_game_result: Option<&CookingMiniGameState>,
    ) -> Result<CookedDish, CookingError> {
        // Validate.
        if ingredient_ids.is_empty() {
            return Err(CookingError::NoIngredients);
        }
        if ingredient_ids.len() > MAX_INGREDIENTS {
            return Err(CookingError::TooManyIngredients);
        }
        if !station.active {
            return Err(CookingError::StationInactive);
        }
        if !station.can_handle(ingredient_ids.len()) {
            return Err(CookingError::StationTooSimple);
        }
        let batch_size = batch_size.clamp(1, station.station_type.max_batch_size().min(MAX_BATCH_SIZE));

        // Check chef skill level for recipe minimum.
        let recipe_id = self.find_matching_recipe(ingredient_ids);
        if let Some(ref rid) = recipe_id {
            if let Some(recipe) = self.recipes.get(rid) {
                if self.chef_skill.level < recipe.min_level {
                    return Err(CookingError::LevelTooLow {
                        required: recipe.min_level,
                        current: self.chef_skill.level,
                    });
                }
            }
        }

        // Collect ingredients.
        let mut ingredients: Vec<CookingIngredient> = Vec::new();
        for id in ingredient_ids {
            match self.ingredients_db.get(id) {
                Some(ing) => {
                    // Check if ingredient is spoiled.
                    if !ing.is_usable() {
                        return Err(CookingError::IngredientSpoiled(id.clone()));
                    }
                    ingredients.push(ing.clone());
                }
                None => return Err(CookingError::UnknownIngredient(id.clone())),
            }
        }

        // Consume fuel.
        if !station.consume_fuel_batch(batch_size) {
            return Err(CookingError::NoFuel);
        }

        // Calculate combined properties.
        let mut combined_props: HashMap<IngredientPropertyType, f32> = HashMap::new();
        let mut avg_quality = 0.0_f32;
        let mut category_synergy_bonus = 0.0f32;
        let mut combination_notes: Vec<String> = Vec::new();

        for ing in &ingredients {
            avg_quality += ing.quality_multiplier();
            for prop in &ing.properties {
                let entry = combined_props.entry(prop.property_type).or_insert(0.0);
                *entry += prop.magnitude * ing.quality_multiplier();
            }
        }
        avg_quality /= ingredients.len() as f32;

        // Calculate category synergies.
        for i in 0..ingredients.len() {
            for j in (i + 1)..ingredients.len() {
                let synergy = ingredients[i].category.category_synergy(&ingredients[j].category);
                if synergy > 0.0 {
                    category_synergy_bonus += synergy;
                    combination_notes.push(format!(
                        "{} and {} complement each other well!",
                        ingredients[i].category.display_name(),
                        ingredients[j].category.display_name(),
                    ));
                }
            }
        }

        // Apply property combination rules.
        let mut quality_modifier = 1.0_f32 + category_synergy_bonus;
        let mut transmuted_props: Vec<(IngredientPropertyType, f32)> = Vec::new();
        let prop_types: Vec<IngredientPropertyType> = combined_props.keys().cloned().collect();

        for i in 0..prop_types.len() {
            for j in (i + 1)..prop_types.len() {
                let mag_a = *combined_props.get(&prop_types[i]).unwrap_or(&0.0);
                let mag_b = *combined_props.get(&prop_types[j]).unwrap_or(&0.0);
                let result = self.rules.evaluate_combination(prop_types[i], mag_a, prop_types[j], mag_b);

                if !result.flavor_description.is_empty() {
                    combination_notes.push(result.flavor_description);
                }

                quality_modifier *= result.quality_modifier;

                if let Some(new) = combined_props.get_mut(&prop_types[i]) {
                    *new = result.magnitude_a_after;
                }
                if let Some(new) = combined_props.get_mut(&prop_types[j]) {
                    *new = result.magnitude_b_after;
                }
                if let Some((prop, mag)) = result.new_property {
                    transmuted_props.push((prop, mag));
                }
            }
        }

        for (prop_type, magnitude) in transmuted_props {
            let entry = combined_props.entry(prop_type).or_insert(0.0);
            *entry += magnitude;
        }

        // Apply chef skill bonuses.
        let chef_bonus = self.chef_skill.quality_bonus()
            + self.chef_skill.specialization_bonus(station.station_type);

        // Apply mastery bonus if using a known recipe.
        let mastery_bonus = recipe_id.as_ref()
            .and_then(|rid| self.recipes.get(rid))
            .map(|r| r.mastery_bonus())
            .unwrap_or(0.0);

        // Apply mini-game bonus.
        let mini_game_bonus = mini_game_result
            .map(|mg| mg.quality_modifier - 1.0)
            .unwrap_or(0.0);

        // Calculate final quality score.
        let station_bonus = station.quality_modifier();
        let quality_score = (avg_quality * quality_modifier + station_bonus + chef_bonus + mastery_bonus + mini_game_bonus)
            .clamp(0.0, 1.0);

        // Chef skill may upgrade effective quality.
        let upgrade_roll = self.next_rng();
        let quality_upgraded = upgrade_roll < self.chef_skill.quality_upgrade_chance();
        let final_quality = if quality_upgraded {
            (quality_score + 0.1).min(1.0)
        } else {
            quality_score
        };

        // Calculate star rating.
        let rating = match final_quality {
            x if x >= 0.9 => 5,
            x if x >= 0.7 => 4,
            x if x >= 0.5 => 3,
            x if x >= 0.3 => 2,
            _ => 1,
        };

        // Generate buffs from properties.
        let mut buffs = Vec::new();
        for (prop_type, magnitude) in &combined_props {
            if prop_type.is_effect() || *prop_type == IngredientPropertyType::Toxic {
                let duration = magnitude * 60.0 * final_quality;
                let buff = FoodBuff::from_property(*prop_type, *magnitude * final_quality, duration);
                buffs.push(buff);
            }
        }

        // Calculate restorations.
        let hp_restore = combined_props
            .get(&IngredientPropertyType::Healing)
            .copied()
            .unwrap_or(0.0)
            * 100.0
            * final_quality;
        let stamina_restore = combined_props
            .get(&IngredientPropertyType::StaminaRestore)
            .copied()
            .unwrap_or(0.0)
            * 100.0
            * final_quality;
        let mana_restore = combined_props
            .get(&IngredientPropertyType::ManaRestore)
            .copied()
            .unwrap_or(0.0)
            * 100.0
            * final_quality;

        // Generate dish name.
        let dish_name = self.generate_dish_name(&ingredients, &combined_props, rating);

        // Record mastery.
        if let Some(ref rid) = recipe_id {
            if let Some(recipe) = self.recipes.get_mut(rid) {
                recipe.record_cook();
            }
        }

        // Check for discovery.
        let ingredient_key = ExperimentationLog::combination_key(ingredient_ids);
        let is_discovery = self.discoveries.insert(ingredient_key);

        // Record in experimentation log.
        self.experimentation.record(
            ingredient_ids,
            &dish_name,
            rating,
            combination_notes.clone(),
            game_time,
        );

        // Discover ingredient properties used.
        for ing_id in ingredient_ids {
            if let Some(ing) = self.ingredients_db.get_mut(ing_id) {
                let undiscovered: Vec<IngredientPropertyType> = ing
                    .properties
                    .iter()
                    .filter(|p| !p.discovered)
                    .map(|p| p.property_type)
                    .collect();
                if let Some(&prop_type) = undiscovered.first() {
                    ing.discover_property(prop_type);
                }
            }
        }

        // Record chef skill.
        self.chef_skill
            .record_cook(station.station_type, rating, is_discovery);

        // Calculate servings (batch size + possible extra from chef skill).
        let extra_roll = self.next_rng();
        let extra = if extra_roll < self.chef_skill.extra_serving_chance() {
            1
        } else {
            0
        };
        let servings = batch_size + extra;

        // Create spoilage state for cooked dish (cooked food lasts longer).
        let cooked_shelf_life = BASE_SPOILAGE_TIME * 2.0;
        let mut spoilage = SpoilageState::new(game_time, cooked_shelf_life, IngredientQuality::Normal);
        if station.station_type.can_preserve() {
            spoilage.preserve();
        }

        // Create the dish.
        let dish = CookedDish {
            item_id: format!(
                "cooked_{}_{}",
                dish_name.to_lowercase().replace(' ', "_"),
                self.history.len()
            ),
            name: dish_name.clone(),
            description: format!(
                "A {}-star dish with {} ingredients ({} servings).",
                rating,
                ingredients.len(),
                servings,
            ),
            rating,
            buffs,
            hp_restore,
            stamina_restore,
            mana_restore,
            sell_value: (rating as u32 * 10 * ingredients.len() as u32 * servings),
            ingredients_used: ingredient_ids.to_vec(),
            recipe_id,
            is_discovery,
            flavor_profile: combined_props
                .into_iter()
                .filter(|(p, _)| p.is_flavor())
                .collect(),
            quality_score: final_quality,
            spoilage,
            servings,
            combination_notes,
        };

        // Record history.
        self.history.push(CookingHistoryEntry {
            dish_name,
            rating,
            was_discovery: is_discovery,
            timestamp: game_time,
            servings,
            mini_game_score: mini_game_result.map(|mg| mg.performance_rating()),
        });

        Ok(dish)
    }

    /// Generate a dish name based on ingredients and properties.
    fn generate_dish_name(
        &self,
        ingredients: &[CookingIngredient],
        properties: &HashMap<IngredientPropertyType, f32>,
        rating: u8,
    ) -> String {
        let prefix = match rating {
            5 => "Exquisite",
            4 => "Delicious",
            3 => "Hearty",
            2 => "Simple",
            _ => "Dubious",
        };

        let main_ingredient = ingredients
            .first()
            .map(|i| i.name.as_str())
            .unwrap_or("Mystery");

        let flavor = properties
            .iter()
            .filter(|(p, _)| p.is_flavor())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(p, _)| match p {
                IngredientPropertyType::Spicy => "Spicy",
                IngredientPropertyType::Sweet => "Sweet",
                IngredientPropertyType::Sour => "Sour",
                IngredientPropertyType::Bitter => "Bitter",
                IngredientPropertyType::Salty => "Salty",
                IngredientPropertyType::Umami => "Savory",
                _ => "",
            })
            .unwrap_or("");

        if flavor.is_empty() {
            format!("{} {} Dish", prefix, main_ingredient)
        } else {
            format!("{} {} {} Dish", prefix, flavor, main_ingredient)
        }
    }

    /// Find a matching recipe for the given ingredients.
    fn find_matching_recipe(&self, ingredient_ids: &[String]) -> Option<String> {
        let mut ingredient_counts: HashMap<&str, u32> = HashMap::new();
        for id in ingredient_ids {
            *ingredient_counts.entry(id).or_insert(0) += 1;
        }

        for (recipe_id, recipe) in &self.recipes {
            if !recipe.unlocked {
                continue;
            }
            let mut matches = true;
            if recipe.ingredients.len() != ingredient_counts.len() {
                continue;
            }
            for (item_id, required) in &recipe.ingredients {
                match ingredient_counts.get(item_id.as_str()) {
                    Some(&have) if have >= *required => {}
                    _ => {
                        matches = false;
                        break;
                    }
                }
            }
            if matches {
                return Some(recipe_id.clone());
            }
        }
        None
    }

    /// Consume a dish and apply its effects.
    pub fn consume_dish(&mut self, dish: &CookedDish) {
        for buff in &dish.buffs {
            if self.active_buffs.len() >= MAX_ACTIVE_FOOD_BUFFS {
                if let Some(idx) = self
                    .active_buffs
                    .iter()
                    .enumerate()
                    .min_by(|a, b| {
                        a.1.remaining_duration
                            .partial_cmp(&b.1.remaining_duration)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    self.active_buffs[idx] = buff.clone();
                    self.active_buffs[idx].activate();
                }
            } else {
                let mut new_buff = buff.clone();
                new_buff.activate();
                self.active_buffs.push(new_buff);
            }
        }
    }

    /// Update active buffs and ingredient spoilage.
    pub fn update(&mut self, dt: f32, game_time: f64) {
        for buff in &mut self.active_buffs {
            buff.update(dt);
        }
        self.active_buffs.retain(|b| b.active);

        // Update spoilage on ingredients.
        for ing in self.ingredients_db.values_mut() {
            ing.update_spoilage(game_time);
        }
    }

    /// Update active buffs only (backward compat).
    pub fn update_buffs(&mut self, dt: f32) {
        for buff in &mut self.active_buffs {
            buff.update(dt);
        }
        self.active_buffs.retain(|b| b.active);
    }

    pub fn active_buffs(&self) -> &[FoodBuff] {
        &self.active_buffs
    }

    pub fn history(&self) -> &[CookingHistoryEntry] {
        &self.history
    }

    pub fn unlocked_recipes(&self) -> Vec<&CookingRecipe> {
        self.recipes.values().filter(|r| r.unlocked).collect()
    }

    pub fn all_recipes(&self) -> Vec<&CookingRecipe> {
        self.recipes.values().collect()
    }

    pub fn stats(&self) -> CookingStats {
        CookingStats {
            total_dishes_cooked: self.history.len(),
            total_discoveries: self.discoveries.len(),
            player_level: self.chef_skill.level,
            player_xp: self.chef_skill.xp,
            xp_for_next_level: self.chef_skill.xp_for_next_level,
            recipes_unlocked: self.recipes.values().filter(|r| r.unlocked).count(),
            total_recipes: self.recipes.len(),
            five_star_dishes: self.history.iter().filter(|h| h.rating == 5).count(),
            active_buffs: self.active_buffs.len(),
            total_experiments: self.experimentation.total_experiments(),
        }
    }
}

/// Cooking statistics.
#[derive(Debug, Clone)]
pub struct CookingStats {
    pub total_dishes_cooked: usize,
    pub total_discoveries: usize,
    pub player_level: u32,
    pub player_xp: u32,
    pub xp_for_next_level: u32,
    pub recipes_unlocked: usize,
    pub total_recipes: usize,
    pub five_star_dishes: usize,
    pub active_buffs: usize,
    pub total_experiments: usize,
}

/// Errors that can occur during cooking.
#[derive(Debug, Clone)]
pub enum CookingError {
    NoIngredients,
    TooManyIngredients,
    StationInactive,
    StationTooSimple,
    UnknownIngredient(String),
    NoFuel,
    LevelTooLow { required: u32, current: u32 },
    MissingIngredient(String),
    IngredientSpoiled(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_ingredient(
        name: &str,
        props: Vec<(IngredientPropertyType, f32)>,
    ) -> CookingIngredient {
        let mut ing = CookingIngredient::new(name, name, IngredientCategory::Vegetable);
        for (prop_type, magnitude) in props {
            ing.properties
                .push(IngredientProperty::discovered(prop_type, magnitude));
        }
        ing
    }

    #[test]
    fn test_ingredient_quality() {
        assert!(IngredientQuality::Legendary.multiplier() > IngredientQuality::Normal.multiplier());
        assert!(IngredientQuality::Poor.multiplier() < IngredientQuality::Normal.multiplier());
    }

    #[test]
    fn test_cooking_station_fuel() {
        let mut station = CookingStation::new(CookingStationType::Campfire);
        station.activate();
        assert!(station.active);
        assert!(station.consume_fuel());
        assert_eq!(station.fuel, 9.0);
    }

    #[test]
    fn test_food_buff_lifecycle() {
        let mut buff = FoodBuff::from_property(IngredientPropertyType::AttackBoost, 0.5, 10.0);
        assert!(!buff.active);
        buff.activate();
        assert!(buff.active);
        buff.update(5.0);
        assert!(buff.active);
        buff.update(6.0);
        assert!(!buff.active);
    }

    #[test]
    fn test_property_combination_rules() {
        let rules = PropertyCombinationRules::default_rules();
        let interaction = rules.get_interaction(
            IngredientPropertyType::Healing,
            IngredientPropertyType::Toxic,
        );
        assert!(matches!(
            interaction,
            PropertyInteraction::Cancellation { .. }
        ));
    }

    #[test]
    fn test_basic_cooking() {
        let mut engine = CookingEngine::new();
        engine.register_ingredient(make_test_ingredient(
            "herb",
            vec![(IngredientPropertyType::Healing, 0.5)],
        ));
        engine.register_ingredient(make_test_ingredient(
            "mushroom",
            vec![(IngredientPropertyType::StaminaRestore, 0.3)],
        ));

        let mut station = CookingStation::new(CookingStationType::Campfire);
        station.activate();

        let result = engine.cook(
            &["herb".to_string(), "mushroom".to_string()],
            &mut station,
            0.0,
        );
        assert!(result.is_ok());
        let dish = result.unwrap();
        assert!(dish.rating >= MIN_RATING);
        assert!(dish.rating <= MAX_RATING);
        assert!(dish.hp_restore > 0.0);
    }

    #[test]
    fn test_spoilage() {
        let mut spoilage = SpoilageState::new(0.0, 100.0, IngredientQuality::Normal);
        assert!(!spoilage.is_spoiled());
        assert_eq!(spoilage.freshness, 1.0);

        spoilage.update(50.0);
        assert!(spoilage.freshness < 1.0);
        assert!(!spoilage.is_spoiled());

        spoilage.update(110.0);
        assert!(spoilage.is_spoiled());
    }

    #[test]
    fn test_spoilage_preservation() {
        let mut spoilage = SpoilageState::new(0.0, 100.0, IngredientQuality::Normal);
        spoilage.preserve();
        spoilage.update(110.0);
        // Should NOT be spoiled because preserved doubles shelf life.
        assert!(!spoilage.is_spoiled());
    }

    #[test]
    fn test_chef_skill_progression() {
        let mut skill = ChefSkill::new();
        assert_eq!(skill.level, 1);

        for _ in 0..20 {
            skill.record_cook(CookingStationType::Campfire, 5, false);
        }
        assert!(skill.level > 1);
        assert!(skill.specializations.get(&CookingStationType::Campfire).copied().unwrap_or(0) > 0);
    }

    #[test]
    fn test_batch_cooking() {
        let mut engine = CookingEngine::new();
        engine.register_ingredient(make_test_ingredient(
            "herb",
            vec![(IngredientPropertyType::Healing, 0.5)],
        ));

        let mut station = CookingStation::new(CookingStationType::Stove);
        station.activate();

        let result = engine.cook_batch(
            &["herb".to_string()],
            &mut station,
            0.0,
            3,
            None,
        );
        assert!(result.is_ok());
        let dish = result.unwrap();
        assert!(dish.servings >= 3);
    }

    #[test]
    fn test_combination_evaluation() {
        let rules = PropertyCombinationRules::default_rules();
        let result = rules.evaluate_combination(
            IngredientPropertyType::Spicy,
            0.5,
            IngredientPropertyType::Sweet,
            0.5,
        );
        assert_eq!(result.interaction_type, "Synergy");
        assert!(result.magnitude_a_after > 0.5);
    }

    #[test]
    fn test_experimentation_log() {
        let mut log = ExperimentationLog::new();
        let ids = vec!["herb".to_string(), "mushroom".to_string()];
        let is_new = log.record(&ids, "Herb Soup", 3, vec![], 0.0);
        assert!(is_new);
        let is_new2 = log.record(&ids, "Herb Soup", 4, vec![], 1.0);
        assert!(!is_new2);
        assert_eq!(log.total_experiments(), 1);
    }

    #[test]
    fn test_quality_grades() {
        assert_eq!(IngredientQuality::Normal.upgrade(), IngredientQuality::Fine);
        assert_eq!(IngredientQuality::Fine.downgrade(), IngredientQuality::Normal);
        assert_eq!(IngredientQuality::Legendary.upgrade(), IngredientQuality::Legendary);
    }

    #[test]
    fn test_category_synergy() {
        let synergy = IngredientCategory::Meat.category_synergy(&IngredientCategory::Spice);
        assert!(synergy > 0.0);
        let no_synergy = IngredientCategory::Meat.category_synergy(&IngredientCategory::Meat);
        assert!((no_synergy).abs() < EPSILON);
    }
}
