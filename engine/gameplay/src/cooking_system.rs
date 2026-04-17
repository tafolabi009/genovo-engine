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

// ---------------------------------------------------------------------------
// IngredientProperty
// ---------------------------------------------------------------------------

/// A property that an ingredient can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IngredientPropertyType {
    /// Heals HP.
    Healing,
    /// Deals damage/poison.
    Toxic,
    /// Spicy flavor.
    Spicy,
    /// Sweet flavor.
    Sweet,
    /// Sour flavor.
    Sour,
    /// Bitter flavor.
    Bitter,
    /// Salty flavor.
    Salty,
    /// Umami flavor.
    Umami,
    /// Increases attack power.
    AttackBoost,
    /// Increases defense.
    DefenseBoost,
    /// Increases speed.
    SpeedBoost,
    /// Restores stamina.
    StaminaRestore,
    /// Restores mana/magic.
    ManaRestore,
    /// Increases fire resistance.
    FireResist,
    /// Increases cold resistance.
    ColdResist,
    /// Increases electricity resistance.
    ElectricResist,
    /// Makes the consumer invisible.
    Stealth,
    /// Increases carrying capacity.
    Strength,
    /// Increases luck.
    Luck,
    /// Energizing (cures tiredness).
    Energizing,
    /// Calming (removes stress/fear).
    Calming,
}

impl IngredientPropertyType {
    /// Get a display name for this property.
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

    /// Check if this is a flavor property.
    pub fn is_flavor(&self) -> bool {
        matches!(
            self,
            Self::Spicy | Self::Sweet | Self::Sour | Self::Bitter | Self::Salty | Self::Umami
        )
    }

    /// Check if this is a buff/effect property.
    pub fn is_effect(&self) -> bool {
        !self.is_flavor() && *self != Self::Toxic
    }
}

/// An ingredient property with a magnitude.
#[derive(Debug, Clone)]
pub struct IngredientProperty {
    /// Type of property.
    pub property_type: IngredientPropertyType,
    /// Magnitude of the property (0.0 to 1.0).
    pub magnitude: f32,
    /// Whether this property has been discovered by the player.
    pub discovered: bool,
}

impl IngredientProperty {
    /// Create a new ingredient property.
    pub fn new(property_type: IngredientPropertyType, magnitude: f32) -> Self {
        Self {
            property_type,
            magnitude: magnitude.clamp(0.0, 1.0),
            discovered: false,
        }
    }

    /// Create a discovered property.
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
    /// Poor quality (stale, damaged).
    Poor,
    /// Normal quality.
    Normal,
    /// Good quality (fresh).
    Good,
    /// Excellent quality (premium).
    Excellent,
    /// Legendary quality (magical/rare).
    Legendary,
}

impl IngredientQuality {
    /// Quality multiplier for recipe rating.
    pub fn multiplier(&self) -> f32 {
        match self {
            Self::Poor => 0.5,
            Self::Normal => 1.0,
            Self::Good => 1.2,
            Self::Excellent => 1.5,
            Self::Legendary => 2.0,
        }
    }

    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Poor => "Poor",
            Self::Normal => "Normal",
            Self::Good => "Good",
            Self::Excellent => "Excellent",
            Self::Legendary => "Legendary",
        }
    }
}

// ---------------------------------------------------------------------------
// Ingredient
// ---------------------------------------------------------------------------

/// An ingredient that can be used in cooking.
#[derive(Debug, Clone)]
pub struct CookingIngredient {
    /// Unique item ID.
    pub item_id: String,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Category (vegetable, meat, spice, herb, mushroom, fruit, mineral, etc.).
    pub category: IngredientCategory,
    /// Properties of this ingredient.
    pub properties: Vec<IngredientProperty>,
    /// Quality tier.
    pub quality: IngredientQuality,
    /// Base value in currency.
    pub base_value: u32,
    /// How many the player has.
    pub quantity: u32,
    /// Whether the ingredient has been fully analyzed (all properties discovered).
    pub fully_analyzed: bool,
    /// Icon reference.
    pub icon: String,
    /// Weight per unit.
    pub weight: f32,
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
    /// Display name.
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
}

impl CookingIngredient {
    /// Create a new ingredient.
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
        }
    }

    /// Add a property.
    pub fn with_property(mut self, prop: IngredientProperty) -> Self {
        self.properties.push(prop);
        self
    }

    /// Set quality.
    pub fn with_quality(mut self, quality: IngredientQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Get all discovered properties.
    pub fn discovered_properties(&self) -> Vec<&IngredientProperty> {
        self.properties.iter().filter(|p| p.discovered).collect()
    }

    /// Discover a property (reveal it to the player).
    pub fn discover_property(&mut self, property_type: IngredientPropertyType) -> bool {
        for prop in &mut self.properties {
            if prop.property_type == property_type && !prop.discovered {
                prop.discovered = true;
                // Check if all properties are now discovered
                self.fully_analyzed = self.properties.iter().all(|p| p.discovered);
                return true;
            }
        }
        false
    }

    /// Discover all properties.
    pub fn discover_all(&mut self) {
        for prop in &mut self.properties {
            prop.discovered = true;
        }
        self.fully_analyzed = true;
    }

    /// Get the magnitude of a specific property.
    pub fn property_magnitude(&self, prop_type: IngredientPropertyType) -> f32 {
        self.properties.iter()
            .find(|p| p.property_type == prop_type)
            .map(|p| p.magnitude)
            .unwrap_or(0.0)
    }

    /// Get the average quality multiplier.
    pub fn quality_multiplier(&self) -> f32 {
        self.quality.multiplier()
    }
}

// ---------------------------------------------------------------------------
// PropertyCombinationRule
// ---------------------------------------------------------------------------

/// How two ingredient properties interact when combined.
#[derive(Debug, Clone)]
pub struct PropertyCombinationRule {
    /// First property type.
    pub prop_a: IngredientPropertyType,
    /// Second property type.
    pub prop_b: IngredientPropertyType,
    /// Interaction type.
    pub interaction: PropertyInteraction,
}

/// How two properties interact.
#[derive(Debug, Clone)]
pub enum PropertyInteraction {
    /// Properties synergize (boosted effect).
    Synergy { multiplier: f32 },
    /// Properties cancel each other out.
    Cancellation { ratio: f32 },
    /// Properties combine into a new effect.
    Transmutation { result: IngredientPropertyType, magnitude: f32 },
    /// One property amplifies the other.
    Amplify { target: IngredientPropertyType, multiplier: f32 },
    /// Properties conflict (bad combination, reduces quality).
    Conflict { quality_penalty: f32 },
    /// No special interaction.
    Neutral,
}

/// Database of combination rules.
#[derive(Debug)]
pub struct PropertyCombinationRules {
    /// Rules indexed by property pair.
    rules: HashMap<(IngredientPropertyType, IngredientPropertyType), PropertyInteraction>,
}

impl PropertyCombinationRules {
    /// Create a new empty rule set.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Create with default alchemy rules.
    pub fn default_rules() -> Self {
        let mut rules = Self::new();

        // Healing + Toxic cancel out
        rules.add_rule(
            IngredientPropertyType::Healing,
            IngredientPropertyType::Toxic,
            PropertyInteraction::Cancellation { ratio: 0.8 },
        );

        // Spicy + Sweet synergy
        rules.add_rule(
            IngredientPropertyType::Spicy,
            IngredientPropertyType::Sweet,
            PropertyInteraction::Synergy { multiplier: 1.3 },
        );

        // Spicy + Sour conflict
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

        // Healing + Calming synergy
        rules.add_rule(
            IngredientPropertyType::Healing,
            IngredientPropertyType::Calming,
            PropertyInteraction::Synergy { multiplier: 1.5 },
        );

        // AttackBoost + SpeedBoost synergy
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

        // Salty + Umami synergy
        rules.add_rule(
            IngredientPropertyType::Salty,
            IngredientPropertyType::Umami,
            PropertyInteraction::Synergy { multiplier: 1.4 },
        );

        // Toxic + Stealth transmutation -> poison invisibility
        rules.add_rule(
            IngredientPropertyType::Toxic,
            IngredientPropertyType::Stealth,
            PropertyInteraction::Transmutation {
                result: IngredientPropertyType::Stealth,
                magnitude: 0.8,
            },
        );

        rules
    }

    /// Add a rule (bidirectional).
    pub fn add_rule(&mut self, a: IngredientPropertyType, b: IngredientPropertyType, interaction: PropertyInteraction) {
        self.rules.insert((a, b), interaction.clone());
        self.rules.insert((b, a), interaction);
    }

    /// Get the interaction between two properties.
    pub fn get_interaction(&self, a: IngredientPropertyType, b: IngredientPropertyType) -> &PropertyInteraction {
        self.rules.get(&(a, b)).unwrap_or(&PropertyInteraction::Neutral)
    }
}

// ---------------------------------------------------------------------------
// CookingStation
// ---------------------------------------------------------------------------

/// Type of cooking station.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CookingStationType {
    /// Basic campfire.
    Campfire,
    /// Kitchen stove.
    Stove,
    /// Oven.
    Oven,
    /// Cauldron (for potions/alchemy).
    Cauldron,
    /// Smoker.
    Smoker,
    /// Grill.
    Grill,
    /// Brewing station (for drinks).
    BrewingStation,
    /// Enchanting table (magical cooking).
    EnchantingTable,
}

impl CookingStationType {
    /// Get the quality bonus for this station type.
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

    /// Get the maximum recipe complexity this station supports.
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

    /// Display name.
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
}

/// A cooking station instance.
#[derive(Debug, Clone)]
pub struct CookingStation {
    /// Station type.
    pub station_type: CookingStationType,
    /// Current fuel level (0.0 to max_fuel).
    pub fuel: f32,
    /// Maximum fuel capacity.
    pub max_fuel: f32,
    /// Fuel consumption per cook.
    pub fuel_per_cook: f32,
    /// Whether the station is lit/active.
    pub active: bool,
    /// Heat level (affects cooking results).
    pub heat: f32,
    /// Maximum heat.
    pub max_heat: f32,
    /// Station level (upgradeable).
    pub level: u32,
    /// Bonus quality from station upgrades.
    pub upgrade_bonus: f32,
}

impl CookingStation {
    /// Create a new cooking station.
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
        }
    }

    /// Add fuel to the station.
    pub fn add_fuel(&mut self, amount: f32) {
        self.fuel = (self.fuel + amount).min(self.max_fuel);
    }

    /// Activate the station (requires fuel).
    pub fn activate(&mut self) -> bool {
        if self.fuel > EPSILON {
            self.active = true;
            self.heat = self.max_heat;
            true
        } else {
            false
        }
    }

    /// Deactivate the station.
    pub fn deactivate(&mut self) {
        self.active = false;
        self.heat = 0.0;
    }

    /// Consume fuel for a cook. Returns false if not enough fuel.
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

    /// Get the total quality modifier from this station.
    pub fn quality_modifier(&self) -> f32 {
        self.station_type.quality_bonus() + self.upgrade_bonus + (self.level as f32 - 1.0) * 0.05
    }

    /// Can this station handle a recipe with N ingredients?
    pub fn can_handle(&self, ingredient_count: usize) -> bool {
        self.active && ingredient_count <= self.station_type.max_complexity()
    }
}

// ---------------------------------------------------------------------------
// FoodBuff
// ---------------------------------------------------------------------------

/// A buff or debuff applied by consuming food.
#[derive(Debug, Clone)]
pub struct FoodBuff {
    /// Unique identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Description of the effect.
    pub description: String,
    /// Property type that this buff is based on.
    pub source_property: IngredientPropertyType,
    /// Magnitude of the effect.
    pub magnitude: f32,
    /// Duration in seconds.
    pub duration: f32,
    /// Whether this is a debuff (negative effect).
    pub is_debuff: bool,
    /// Icon reference.
    pub icon: String,
    /// Remaining duration (when active).
    pub remaining_duration: f32,
    /// Whether this buff is currently active.
    pub active: bool,
}

impl FoodBuff {
    /// Create a new food buff from a property.
    pub fn from_property(property: IngredientPropertyType, magnitude: f32, duration: f32) -> Self {
        let is_debuff = matches!(property, IngredientPropertyType::Toxic);
        Self {
            id: format!("food_{:?}_{}", property, (magnitude * 100.0) as u32),
            name: format!("{} {}", property.display_name(), if is_debuff { "Debuff" } else { "Buff" }),
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

    /// Activate this buff.
    pub fn activate(&mut self) {
        self.active = true;
        self.remaining_duration = self.duration;
    }

    /// Update the buff (tick duration).
    pub fn update(&mut self, dt: f32) {
        if self.active {
            self.remaining_duration -= dt;
            if self.remaining_duration <= 0.0 {
                self.active = false;
                self.remaining_duration = 0.0;
            }
        }
    }

    /// Get the remaining fraction (0..1).
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
    /// Generated item ID.
    pub item_id: String,
    /// Name of the dish.
    pub name: String,
    /// Description.
    pub description: String,
    /// Star rating (1-5).
    pub rating: u8,
    /// Buffs granted when consumed.
    pub buffs: Vec<FoodBuff>,
    /// HP restored on consumption.
    pub hp_restore: f32,
    /// Stamina restored.
    pub stamina_restore: f32,
    /// Mana restored.
    pub mana_restore: f32,
    /// Sale value.
    pub sell_value: u32,
    /// Ingredients used.
    pub ingredients_used: Vec<String>,
    /// Recipe ID (if a known recipe was used).
    pub recipe_id: Option<String>,
    /// Whether this was a new discovery.
    pub is_discovery: bool,
    /// Flavor profile (dominant flavors).
    pub flavor_profile: HashMap<IngredientPropertyType, f32>,
    /// Quality score (0..1).
    pub quality_score: f32,
}

impl CookedDish {
    /// Get the dominant flavor.
    pub fn dominant_flavor(&self) -> Option<IngredientPropertyType> {
        self.flavor_profile.iter()
            .filter(|(prop, _)| prop.is_flavor())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(prop, _)| *prop)
    }

    /// Get total buff count.
    pub fn buff_count(&self) -> usize {
        self.buffs.iter().filter(|b| !b.is_debuff).count()
    }

    /// Get total debuff count.
    pub fn debuff_count(&self) -> usize {
        self.buffs.iter().filter(|b| b.is_debuff).count()
    }
}

// ---------------------------------------------------------------------------
// CookingRecipe
// ---------------------------------------------------------------------------

/// A known recipe for cooking.
#[derive(Debug, Clone)]
pub struct CookingRecipe {
    /// Unique recipe ID.
    pub id: String,
    /// Recipe name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Required ingredients (item_id -> quantity).
    pub ingredients: HashMap<String, u32>,
    /// Required station type.
    pub station_type: CookingStationType,
    /// Minimum player cooking level.
    pub min_level: u32,
    /// Base quality score for this recipe.
    pub base_quality: f32,
    /// Whether the recipe is unlocked.
    pub unlocked: bool,
    /// Number of times this recipe has been cooked.
    pub times_cooked: u32,
    /// Mastery level (0..MAX_MASTERY_LEVEL).
    pub mastery_level: u32,
    /// Resulting dish name.
    pub result_name: String,
    /// Resulting dish description.
    pub result_description: String,
    /// Category tags.
    pub tags: Vec<String>,
}

impl CookingRecipe {
    /// Create a new recipe.
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
        }
    }

    /// Add an ingredient requirement.
    pub fn with_ingredient(mut self, item_id: impl Into<String>, quantity: u32) -> Self {
        self.ingredients.insert(item_id.into(), quantity);
        self
    }

    /// Set station type.
    pub fn with_station(mut self, station: CookingStationType) -> Self {
        self.station_type = station;
        self
    }

    /// Record a successful cook.
    pub fn record_cook(&mut self) {
        self.times_cooked += 1;
        let new_mastery = (self.times_cooked / COOKS_PER_MASTERY).min(MAX_MASTERY_LEVEL);
        if new_mastery > self.mastery_level {
            self.mastery_level = new_mastery;
        }
    }

    /// Get the mastery quality bonus.
    pub fn mastery_bonus(&self) -> f32 {
        self.mastery_level as f32 * 0.1
    }

    /// Check if the player has the required ingredients.
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
}

// ---------------------------------------------------------------------------
// CookingEngine
// ---------------------------------------------------------------------------

/// The main cooking engine that produces dishes from ingredients.
pub struct CookingEngine {
    /// Property combination rules.
    pub rules: PropertyCombinationRules,
    /// Ingredient database.
    ingredients_db: HashMap<String, CookingIngredient>,
    /// Recipe book.
    recipes: HashMap<String, CookingRecipe>,
    /// Player cooking level.
    pub player_level: u32,
    /// Player cooking XP.
    pub player_xp: u32,
    /// XP needed for next level.
    pub xp_for_next_level: u32,
    /// Active food buffs on the player.
    active_buffs: Vec<FoodBuff>,
    /// Cooking history.
    history: Vec<CookingHistoryEntry>,
    /// Discovery log.
    discoveries: HashSet<String>,
}

/// Entry in the cooking history.
#[derive(Debug, Clone)]
pub struct CookingHistoryEntry {
    /// Dish that was cooked.
    pub dish_name: String,
    /// Rating achieved.
    pub rating: u8,
    /// Whether it was a new discovery.
    pub was_discovery: bool,
    /// Timestamp (game time).
    pub timestamp: f64,
}

impl CookingEngine {
    /// Create a new cooking engine.
    pub fn new() -> Self {
        Self {
            rules: PropertyCombinationRules::default_rules(),
            ingredients_db: HashMap::new(),
            recipes: HashMap::new(),
            player_level: 1,
            player_xp: 0,
            xp_for_next_level: 100,
            active_buffs: Vec::new(),
            history: Vec::new(),
            discoveries: HashSet::new(),
        }
    }

    /// Register an ingredient in the database.
    pub fn register_ingredient(&mut self, ingredient: CookingIngredient) {
        self.ingredients_db.insert(ingredient.item_id.clone(), ingredient);
    }

    /// Register a recipe.
    pub fn register_recipe(&mut self, recipe: CookingRecipe) {
        self.recipes.insert(recipe.id.clone(), recipe);
    }

    /// Unlock a recipe.
    pub fn unlock_recipe(&mut self, recipe_id: &str) -> bool {
        if let Some(recipe) = self.recipes.get_mut(recipe_id) {
            recipe.unlocked = true;
            true
        } else {
            false
        }
    }

    /// Cook a dish using the given ingredients at the given station.
    pub fn cook(
        &mut self,
        ingredient_ids: &[String],
        station: &mut CookingStation,
        game_time: f64,
    ) -> Result<CookedDish, CookingError> {
        // Validate
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

        // Collect ingredients
        let mut ingredients: Vec<CookingIngredient> = Vec::new();
        for id in ingredient_ids {
            match self.ingredients_db.get(id) {
                Some(ing) => ingredients.push(ing.clone()),
                None => return Err(CookingError::UnknownIngredient(id.clone())),
            }
        }

        // Consume fuel
        if !station.consume_fuel() {
            return Err(CookingError::NoFuel);
        }

        // Calculate combined properties
        let mut combined_props: HashMap<IngredientPropertyType, f32> = HashMap::new();
        let mut avg_quality = 0.0_f32;

        for ing in &ingredients {
            avg_quality += ing.quality_multiplier();
            for prop in &ing.properties {
                let entry = combined_props.entry(prop.property_type).or_insert(0.0);
                *entry += prop.magnitude * ing.quality_multiplier();
            }
        }
        avg_quality /= ingredients.len() as f32;

        // Apply combination rules
        let mut quality_modifier = 1.0_f32;
        let mut transmuted_props: Vec<(IngredientPropertyType, f32)> = Vec::new();
        let prop_types: Vec<IngredientPropertyType> = combined_props.keys().cloned().collect();

        for i in 0..prop_types.len() {
            for j in (i + 1)..prop_types.len() {
                let interaction = self.rules.get_interaction(prop_types[i], prop_types[j]);
                match interaction {
                    PropertyInteraction::Synergy { multiplier } => {
                        if let Some(val) = combined_props.get_mut(&prop_types[i]) {
                            *val *= multiplier;
                        }
                        if let Some(val) = combined_props.get_mut(&prop_types[j]) {
                            *val *= multiplier;
                        }
                    }
                    PropertyInteraction::Cancellation { ratio } => {
                        if let Some(val) = combined_props.get_mut(&prop_types[i]) {
                            *val *= 1.0 - ratio;
                        }
                        if let Some(val) = combined_props.get_mut(&prop_types[j]) {
                            *val *= 1.0 - ratio;
                        }
                    }
                    PropertyInteraction::Transmutation { result, magnitude } => {
                        transmuted_props.push((*result, *magnitude));
                    }
                    PropertyInteraction::Amplify { target, multiplier } => {
                        if let Some(val) = combined_props.get_mut(target) {
                            *val *= multiplier;
                        }
                    }
                    PropertyInteraction::Conflict { quality_penalty } => {
                        quality_modifier -= quality_penalty;
                    }
                    PropertyInteraction::Neutral => {}
                }
            }
        }

        // Add transmuted properties
        for (prop_type, magnitude) in transmuted_props {
            let entry = combined_props.entry(prop_type).or_insert(0.0);
            *entry += magnitude;
        }

        // Calculate final quality score
        let station_bonus = station.quality_modifier();
        let quality_score = (avg_quality * quality_modifier + station_bonus).clamp(0.0, 1.0);

        // Calculate star rating
        let rating = match quality_score {
            x if x >= 0.9 => 5,
            x if x >= 0.7 => 4,
            x if x >= 0.5 => 3,
            x if x >= 0.3 => 2,
            _ => 1,
        };

        // Generate buffs from properties
        let mut buffs = Vec::new();
        for (prop_type, magnitude) in &combined_props {
            if prop_type.is_effect() || *prop_type == IngredientPropertyType::Toxic {
                let duration = magnitude * 60.0 * quality_score;
                let buff = FoodBuff::from_property(*prop_type, *magnitude * quality_score, duration);
                buffs.push(buff);
            }
        }

        // Calculate restorations
        let hp_restore = combined_props.get(&IngredientPropertyType::Healing)
            .copied().unwrap_or(0.0) * 100.0 * quality_score;
        let stamina_restore = combined_props.get(&IngredientPropertyType::StaminaRestore)
            .copied().unwrap_or(0.0) * 100.0 * quality_score;
        let mana_restore = combined_props.get(&IngredientPropertyType::ManaRestore)
            .copied().unwrap_or(0.0) * 100.0 * quality_score;

        // Generate dish name
        let dish_name = self.generate_dish_name(&ingredients, &combined_props, rating);

        // Check for recipe match
        let recipe_id = self.find_matching_recipe(ingredient_ids);
        if let Some(ref rid) = recipe_id {
            if let Some(recipe) = self.recipes.get_mut(rid) {
                recipe.record_cook();
            }
        }

        // Check for discovery
        let ingredient_key: String = {
            let mut sorted: Vec<_> = ingredient_ids.to_vec();
            sorted.sort();
            sorted.join("+")
        };
        let is_discovery = self.discoveries.insert(ingredient_key);

        // Discover ingredient properties used
        for ing_id in ingredient_ids {
            if let Some(ing) = self.ingredients_db.get_mut(ing_id) {
                // Discover one random undiscovered property
                let undiscovered: Vec<IngredientPropertyType> = ing.properties.iter()
                    .filter(|p| !p.discovered)
                    .map(|p| p.property_type)
                    .collect();
                if let Some(&prop_type) = undiscovered.first() {
                    ing.discover_property(prop_type);
                }
            }
        }

        // Award XP
        let mut xp = XP_PER_COOK;
        if rating == MAX_RATING {
            xp += XP_BONUS_FIVE_STAR;
        }
        if is_discovery {
            xp += 25;
        }
        self.add_xp(xp);

        // Create the dish
        let dish = CookedDish {
            item_id: format!("cooked_{}_{}", dish_name.to_lowercase().replace(' ', "_"), self.history.len()),
            name: dish_name.clone(),
            description: format!("A {}-star dish with {} ingredients.", rating, ingredients.len()),
            rating,
            buffs,
            hp_restore,
            stamina_restore,
            mana_restore,
            sell_value: (rating as u32 * 10 * ingredients.len() as u32),
            ingredients_used: ingredient_ids.to_vec(),
            recipe_id,
            is_discovery,
            flavor_profile: combined_props.into_iter()
                .filter(|(p, _)| p.is_flavor())
                .collect(),
            quality_score,
        };

        // Record history
        self.history.push(CookingHistoryEntry {
            dish_name,
            rating,
            was_discovery: is_discovery,
            timestamp: game_time,
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

        let main_ingredient = ingredients.first()
            .map(|i| i.name.as_str())
            .unwrap_or("Mystery");

        // Find dominant flavor
        let flavor = properties.iter()
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
                    _ => { matches = false; break; }
                }
            }
            if matches {
                return Some(recipe_id.clone());
            }
        }
        None
    }

    /// Add XP and check for level up.
    fn add_xp(&mut self, xp: u32) {
        self.player_xp += xp;
        while self.player_xp >= self.xp_for_next_level {
            self.player_xp -= self.xp_for_next_level;
            self.player_level += 1;
            self.xp_for_next_level = self.player_level * 100 + 50;
        }
    }

    /// Consume a dish and apply its effects.
    pub fn consume_dish(&mut self, dish: &CookedDish) {
        // Apply buffs (limited to MAX_ACTIVE_FOOD_BUFFS)
        for buff in &dish.buffs {
            if self.active_buffs.len() >= MAX_ACTIVE_FOOD_BUFFS {
                // Replace the buff with least remaining time
                if let Some(idx) = self.active_buffs.iter()
                    .enumerate()
                    .min_by(|a, b| a.1.remaining_duration.partial_cmp(&b.1.remaining_duration).unwrap_or(std::cmp::Ordering::Equal))
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

    /// Update active buffs.
    pub fn update_buffs(&mut self, dt: f32) {
        for buff in &mut self.active_buffs {
            buff.update(dt);
        }
        self.active_buffs.retain(|b| b.active);
    }

    /// Get active buffs.
    pub fn active_buffs(&self) -> &[FoodBuff] {
        &self.active_buffs
    }

    /// Get cooking history.
    pub fn history(&self) -> &[CookingHistoryEntry] {
        &self.history
    }

    /// Get unlocked recipes.
    pub fn unlocked_recipes(&self) -> Vec<&CookingRecipe> {
        self.recipes.values().filter(|r| r.unlocked).collect()
    }

    /// Get all recipes.
    pub fn all_recipes(&self) -> Vec<&CookingRecipe> {
        self.recipes.values().collect()
    }

    /// Get statistics.
    pub fn stats(&self) -> CookingStats {
        CookingStats {
            total_dishes_cooked: self.history.len(),
            total_discoveries: self.discoveries.len(),
            player_level: self.player_level,
            player_xp: self.player_xp,
            xp_for_next_level: self.xp_for_next_level,
            recipes_unlocked: self.recipes.values().filter(|r| r.unlocked).count(),
            total_recipes: self.recipes.len(),
            five_star_dishes: self.history.iter().filter(|h| h.rating == 5).count(),
            active_buffs: self.active_buffs.len(),
        }
    }
}

/// Cooking statistics.
#[derive(Debug, Clone)]
pub struct CookingStats {
    /// Total dishes cooked.
    pub total_dishes_cooked: usize,
    /// Total unique ingredient combinations discovered.
    pub total_discoveries: usize,
    /// Player cooking level.
    pub player_level: u32,
    /// Current XP.
    pub player_xp: u32,
    /// XP needed for next level.
    pub xp_for_next_level: u32,
    /// Number of unlocked recipes.
    pub recipes_unlocked: usize,
    /// Total number of recipes.
    pub total_recipes: usize,
    /// Number of 5-star dishes cooked.
    pub five_star_dishes: usize,
    /// Number of active food buffs.
    pub active_buffs: usize,
}

/// Errors that can occur during cooking.
#[derive(Debug, Clone)]
pub enum CookingError {
    /// No ingredients provided.
    NoIngredients,
    /// Too many ingredients.
    TooManyIngredients,
    /// Cooking station is not active.
    StationInactive,
    /// Station cannot handle this many ingredients.
    StationTooSimple,
    /// Unknown ingredient ID.
    UnknownIngredient(String),
    /// Not enough fuel.
    NoFuel,
    /// Player level too low.
    LevelTooLow { required: u32, current: u32 },
    /// Missing required ingredient.
    MissingIngredient(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_ingredient(name: &str, props: Vec<(IngredientPropertyType, f32)>) -> CookingIngredient {
        let mut ing = CookingIngredient::new(name, name, IngredientCategory::Vegetable);
        for (prop_type, magnitude) in props {
            ing.properties.push(IngredientProperty::discovered(prop_type, magnitude));
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
        let mut buff = FoodBuff::from_property(
            IngredientPropertyType::AttackBoost,
            0.5,
            10.0,
        );
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
        assert!(matches!(interaction, PropertyInteraction::Cancellation { .. }));
    }

    #[test]
    fn test_basic_cooking() {
        let mut engine = CookingEngine::new();
        engine.register_ingredient(make_test_ingredient("herb", vec![
            (IngredientPropertyType::Healing, 0.5),
        ]));
        engine.register_ingredient(make_test_ingredient("mushroom", vec![
            (IngredientPropertyType::StaminaRestore, 0.3),
        ]));

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
}
