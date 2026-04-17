//! Advanced crafting system (v2).
//!
//! Provides recipe discovery, categorized recipe books, crafting station
//! requirements, quality tiers from common to legendary, random stat rolls,
//! material bonuses, durability from materials, and an enchanting/modification
//! system.
//!
//! # Key concepts
//!
//! - **Recipe**: A formula for turning inputs into outputs, with optional
//!   station requirements and skill checks.
//! - **RecipeCategory**: Grouping for recipes (Weaponsmithing, Armorsmithing,
//!   Alchemy, Cooking, Enchanting, etc.).
//! - **CraftingStation**: A specific workbench or tool required for crafting.
//! - **QualityTier**: Output quality from Common to Legendary, affected by
//!   skill, materials, and random rolls.
//! - **MaterialBonus**: Bonus stats granted by using specific material types.
//! - **Enchantment**: A modification that adds or enhances properties on
//!   crafted items.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum recipes in a recipe book.
pub const MAX_RECIPES: usize = 1024;

/// Maximum ingredients per recipe.
pub const MAX_INGREDIENTS: usize = 8;

/// Maximum outputs per recipe.
pub const MAX_OUTPUTS: usize = 4;

/// Maximum enchantment slots per item.
pub const MAX_ENCHANT_SLOTS: usize = 4;

/// Maximum number of crafting stations.
pub const MAX_STATIONS: usize = 64;

/// Maximum number of material bonuses per material.
pub const MAX_MATERIAL_BONUSES: usize = 8;

/// Base chance for quality upgrade (0..1).
pub const BASE_QUALITY_CHANCE: f32 = 0.1;

/// Skill contribution to quality chance.
pub const SKILL_QUALITY_FACTOR: f32 = 0.005;

/// Maximum stat roll variance (fraction of base stat).
pub const MAX_STAT_VARIANCE: f32 = 0.2;

/// Minimum durability multiplier from materials.
pub const MIN_DURABILITY_MULT: f32 = 0.5;

/// Maximum durability multiplier from materials.
pub const MAX_DURABILITY_MULT: f32 = 3.0;

/// Enchanting success base chance.
pub const ENCHANT_BASE_CHANCE: f32 = 0.8;

/// Enchanting failure penalty (durability loss fraction).
pub const ENCHANT_FAILURE_PENALTY: f32 = 0.1;

/// Maximum number of modification passes on a single item.
pub const MAX_MODIFICATIONS: usize = 8;

// ---------------------------------------------------------------------------
// QualityTier
// ---------------------------------------------------------------------------

/// Quality level of a crafted item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QualityTier {
    /// Basic quality.
    Common,
    /// Above average.
    Uncommon,
    /// Clearly superior.
    Rare,
    /// Exceptional quality.
    Epic,
    /// Peak of mortal craftsmanship.
    Legendary,
}

impl QualityTier {
    /// Stat multiplier for this quality tier.
    pub fn stat_multiplier(&self) -> f32 {
        match self {
            Self::Common => 1.0,
            Self::Uncommon => 1.15,
            Self::Rare => 1.35,
            Self::Epic => 1.6,
            Self::Legendary => 2.0,
        }
    }

    /// Number of enchantment slots.
    pub fn enchant_slots(&self) -> usize {
        match self {
            Self::Common => 1,
            Self::Uncommon => 1,
            Self::Rare => 2,
            Self::Epic => 3,
            Self::Legendary => 4,
        }
    }

    /// Durability multiplier.
    pub fn durability_multiplier(&self) -> f32 {
        match self {
            Self::Common => 1.0,
            Self::Uncommon => 1.1,
            Self::Rare => 1.25,
            Self::Epic => 1.5,
            Self::Legendary => 2.0,
        }
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Common => "Common",
            Self::Uncommon => "Uncommon",
            Self::Rare => "Rare",
            Self::Epic => "Epic",
            Self::Legendary => "Legendary",
        }
    }

    /// Color hex for UI.
    pub fn color_hex(&self) -> &'static str {
        match self {
            Self::Common => "#FFFFFF",
            Self::Uncommon => "#1EFF00",
            Self::Rare => "#0070DD",
            Self::Epic => "#A335EE",
            Self::Legendary => "#FF8000",
        }
    }

    /// Try to upgrade to the next tier. Returns None if already max.
    pub fn upgrade(&self) -> Option<QualityTier> {
        match self {
            Self::Common => Some(Self::Uncommon),
            Self::Uncommon => Some(Self::Rare),
            Self::Rare => Some(Self::Epic),
            Self::Epic => Some(Self::Legendary),
            Self::Legendary => None,
        }
    }

    /// Numeric index (0-4).
    pub fn index(&self) -> usize {
        match self {
            Self::Common => 0,
            Self::Uncommon => 1,
            Self::Rare => 2,
            Self::Epic => 3,
            Self::Legendary => 4,
        }
    }
}

impl Default for QualityTier {
    fn default() -> Self {
        Self::Common
    }
}

// ---------------------------------------------------------------------------
// RecipeCategory
// ---------------------------------------------------------------------------

/// Category for grouping recipes in the crafting UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecipeCategory {
    /// Sword, axe, mace, dagger, bow.
    Weaponsmithing,
    /// Helmets, chest, legs, gloves, boots.
    Armorsmithing,
    /// Potions, elixirs, poisons.
    Alchemy,
    /// Food and drink with buffs.
    Cooking,
    /// Magical modifications and rune crafting.
    Enchanting,
    /// Jewelry: rings, necklaces, amulets.
    Jewelcrafting,
    /// Leather working and cloth armor.
    Tailoring,
    /// Carpentry: bows, staves, furniture.
    Woodworking,
    /// Mining refinement: ore to ingots.
    Smelting,
    /// General utility items.
    General,
}

impl RecipeCategory {
    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Weaponsmithing => "Weaponsmithing",
            Self::Armorsmithing => "Armorsmithing",
            Self::Alchemy => "Alchemy",
            Self::Cooking => "Cooking",
            Self::Enchanting => "Enchanting",
            Self::Jewelcrafting => "Jewelcrafting",
            Self::Tailoring => "Tailoring",
            Self::Woodworking => "Woodworking",
            Self::Smelting => "Smelting",
            Self::General => "General",
        }
    }

    /// Associated skill name.
    pub fn skill_name(&self) -> &'static str {
        match self {
            Self::Weaponsmithing => "weaponsmithing",
            Self::Armorsmithing => "armorsmithing",
            Self::Alchemy => "alchemy",
            Self::Cooking => "cooking",
            Self::Enchanting => "enchanting",
            Self::Jewelcrafting => "jewelcrafting",
            Self::Tailoring => "tailoring",
            Self::Woodworking => "woodworking",
            Self::Smelting => "smelting",
            Self::General => "general_crafting",
        }
    }
}

// ---------------------------------------------------------------------------
// CraftingStation
// ---------------------------------------------------------------------------

/// A crafting station or workbench.
#[derive(Debug, Clone)]
pub struct CraftingStation {
    /// Unique station type ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Which recipe categories this station supports.
    pub supported_categories: Vec<RecipeCategory>,
    /// Station tier (higher tier = access to better recipes).
    pub tier: u32,
    /// Quality bonus for crafting at this station.
    pub quality_bonus: f32,
    /// Speed multiplier (1.0 = normal, 2.0 = twice as fast).
    pub speed_multiplier: f32,
    /// Whether the station requires fuel.
    pub requires_fuel: bool,
    /// Fuel item ID (if requires_fuel).
    pub fuel_item: Option<String>,
    /// Fuel consumption rate per craft.
    pub fuel_per_craft: u32,
}

impl CraftingStation {
    /// Create a new crafting station.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        categories: Vec<RecipeCategory>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            supported_categories: categories,
            tier: 1,
            quality_bonus: 0.0,
            speed_multiplier: 1.0,
            requires_fuel: false,
            fuel_item: None,
            fuel_per_craft: 0,
        }
    }

    /// Set the station tier.
    pub fn with_tier(mut self, tier: u32) -> Self {
        self.tier = tier;
        self
    }

    /// Set a quality bonus.
    pub fn with_quality_bonus(mut self, bonus: f32) -> Self {
        self.quality_bonus = bonus;
        self
    }

    /// Set a speed multiplier.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed_multiplier = speed;
        self
    }

    /// Make the station require fuel.
    pub fn with_fuel(mut self, fuel_item: impl Into<String>, per_craft: u32) -> Self {
        self.requires_fuel = true;
        self.fuel_item = Some(fuel_item.into());
        self.fuel_per_craft = per_craft;
        self
    }

    /// Check if this station supports a category.
    pub fn supports(&self, category: RecipeCategory) -> bool {
        self.supported_categories.contains(&category)
    }
}

// ---------------------------------------------------------------------------
// Ingredient
// ---------------------------------------------------------------------------

/// A single ingredient in a recipe.
#[derive(Debug, Clone)]
pub struct Ingredient {
    /// Item ID.
    pub item_id: String,
    /// Quantity required.
    pub quantity: u32,
    /// Whether this ingredient is consumed (some are tools/catalysts).
    pub consumed: bool,
    /// Optional: acceptable substitute item IDs.
    pub substitutes: Vec<String>,
}

impl Ingredient {
    /// Create a new consumable ingredient.
    pub fn new(item_id: impl Into<String>, quantity: u32) -> Self {
        Self {
            item_id: item_id.into(),
            quantity,
            consumed: true,
            substitutes: Vec::new(),
        }
    }

    /// Mark as a catalyst (not consumed).
    pub fn as_catalyst(mut self) -> Self {
        self.consumed = false;
        self
    }

    /// Add a substitute item.
    pub fn or(mut self, substitute: impl Into<String>) -> Self {
        self.substitutes.push(substitute.into());
        self
    }

    /// Check if an item ID satisfies this ingredient.
    pub fn accepts(&self, item_id: &str) -> bool {
        self.item_id == item_id || self.substitutes.iter().any(|s| s == item_id)
    }
}

// ---------------------------------------------------------------------------
// CraftingOutput
// ---------------------------------------------------------------------------

/// An output produced by a recipe.
#[derive(Debug, Clone)]
pub struct CraftingOutput {
    /// Item ID to produce.
    pub item_id: String,
    /// Base quantity produced.
    pub quantity: u32,
    /// Chance to produce (0..1, 1.0 = guaranteed).
    pub chance: f32,
    /// Base stats for the produced item.
    pub base_stats: HashMap<String, f32>,
    /// Base durability.
    pub base_durability: f32,
}

impl CraftingOutput {
    /// Create a new guaranteed output.
    pub fn new(item_id: impl Into<String>, quantity: u32) -> Self {
        Self {
            item_id: item_id.into(),
            quantity,
            chance: 1.0,
            base_stats: HashMap::new(),
            base_durability: 100.0,
        }
    }

    /// Set the production chance.
    pub fn with_chance(mut self, chance: f32) -> Self {
        self.chance = chance.clamp(0.0, 1.0);
        self
    }

    /// Add a base stat.
    pub fn with_stat(mut self, stat: impl Into<String>, value: f32) -> Self {
        self.base_stats.insert(stat.into(), value);
        self
    }

    /// Set base durability.
    pub fn with_durability(mut self, durability: f32) -> Self {
        self.base_durability = durability;
        self
    }
}

// ---------------------------------------------------------------------------
// Recipe
// ---------------------------------------------------------------------------

/// A crafting recipe.
#[derive(Debug, Clone)]
pub struct Recipe {
    /// Unique recipe ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Category.
    pub category: RecipeCategory,
    /// Required ingredients.
    pub ingredients: Vec<Ingredient>,
    /// Outputs produced.
    pub outputs: Vec<CraftingOutput>,
    /// Required crafting station type (None = hand-crafted).
    pub required_station: Option<String>,
    /// Minimum station tier required.
    pub min_station_tier: u32,
    /// Required skill and minimum level.
    pub skill_requirement: Option<(String, u32)>,
    /// Crafting time in seconds.
    pub craft_time: f32,
    /// Whether this recipe must be discovered first.
    pub requires_discovery: bool,
    /// XP awarded for crafting.
    pub xp_reward: f32,
    /// Minimum quality tier achievable.
    pub min_quality: QualityTier,
    /// Maximum quality tier achievable.
    pub max_quality: QualityTier,
    /// Difficulty modifier for quality rolls.
    pub difficulty: f32,
}

impl Recipe {
    /// Create a new recipe.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        category: RecipeCategory,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            category,
            ingredients: Vec::new(),
            outputs: Vec::new(),
            required_station: None,
            min_station_tier: 1,
            skill_requirement: None,
            craft_time: 5.0,
            requires_discovery: false,
            xp_reward: 10.0,
            min_quality: QualityTier::Common,
            max_quality: QualityTier::Legendary,
            difficulty: 1.0,
        }
    }

    /// Add an ingredient.
    pub fn ingredient(mut self, ingredient: Ingredient) -> Self {
        if self.ingredients.len() < MAX_INGREDIENTS {
            self.ingredients.push(ingredient);
        }
        self
    }

    /// Add an output.
    pub fn output(mut self, output: CraftingOutput) -> Self {
        if self.outputs.len() < MAX_OUTPUTS {
            self.outputs.push(output);
        }
        self
    }

    /// Require a crafting station.
    pub fn at_station(mut self, station_id: impl Into<String>) -> Self {
        self.required_station = Some(station_id.into());
        self
    }

    /// Set minimum station tier.
    pub fn min_tier(mut self, tier: u32) -> Self {
        self.min_station_tier = tier;
        self
    }

    /// Require a skill level.
    pub fn requires_skill(mut self, skill: impl Into<String>, level: u32) -> Self {
        self.skill_requirement = Some((skill.into(), level));
        self
    }

    /// Set craft time.
    pub fn craft_time(mut self, seconds: f32) -> Self {
        self.craft_time = seconds;
        self
    }

    /// Require discovery before use.
    pub fn must_discover(mut self) -> Self {
        self.requires_discovery = true;
        self
    }

    /// Set XP reward.
    pub fn xp(mut self, amount: f32) -> Self {
        self.xp_reward = amount;
        self
    }

    /// Set quality range.
    pub fn quality_range(mut self, min: QualityTier, max: QualityTier) -> Self {
        self.min_quality = min;
        self.max_quality = max;
        self
    }

    /// Set difficulty.
    pub fn difficulty(mut self, diff: f32) -> Self {
        self.difficulty = diff;
        self
    }

    /// Check if the recipe can be crafted given available ingredients.
    pub fn can_craft(&self, available: &HashMap<String, u32>) -> bool {
        for ingredient in &self.ingredients {
            let needed = ingredient.quantity;
            let have = available.get(&ingredient.item_id).copied().unwrap_or(0);

            if have < needed {
                // Check substitutes
                let sub_ok = ingredient.substitutes.iter().any(|sub| {
                    available.get(sub).copied().unwrap_or(0) >= needed
                });
                if !sub_ok {
                    return false;
                }
            }
        }
        true
    }

    /// Get missing ingredients.
    pub fn missing_ingredients(
        &self,
        available: &HashMap<String, u32>,
    ) -> Vec<(String, u32)> {
        let mut missing = Vec::new();
        for ingredient in &self.ingredients {
            let needed = ingredient.quantity;
            let have = available.get(&ingredient.item_id).copied().unwrap_or(0);
            if have < needed {
                missing.push((ingredient.item_id.clone(), needed - have));
            }
        }
        missing
    }
}

// ---------------------------------------------------------------------------
// MaterialBonus
// ---------------------------------------------------------------------------

/// Bonus stats granted by using a specific material.
#[derive(Debug, Clone)]
pub struct MaterialBonus {
    /// Material item ID.
    pub material_id: String,
    /// Display name for the material.
    pub material_name: String,
    /// Stat bonuses: stat_name -> bonus_value.
    pub stat_bonuses: HashMap<String, f32>,
    /// Durability multiplier.
    pub durability_multiplier: f32,
    /// Weight modifier.
    pub weight_modifier: f32,
    /// Visual variant name.
    pub visual_variant: String,
}

impl MaterialBonus {
    /// Create a new material bonus.
    pub fn new(
        material_id: impl Into<String>,
        material_name: impl Into<String>,
    ) -> Self {
        Self {
            material_id: material_id.into(),
            material_name: material_name.into(),
            stat_bonuses: HashMap::new(),
            durability_multiplier: 1.0,
            weight_modifier: 1.0,
            visual_variant: String::new(),
        }
    }

    /// Add a stat bonus.
    pub fn with_stat(mut self, stat: impl Into<String>, bonus: f32) -> Self {
        self.stat_bonuses.insert(stat.into(), bonus);
        self
    }

    /// Set durability multiplier.
    pub fn with_durability(mut self, mult: f32) -> Self {
        self.durability_multiplier = mult.clamp(MIN_DURABILITY_MULT, MAX_DURABILITY_MULT);
        self
    }

    /// Set weight modifier.
    pub fn with_weight(mut self, modifier: f32) -> Self {
        self.weight_modifier = modifier;
        self
    }

    /// Set visual variant.
    pub fn with_visual(mut self, variant: impl Into<String>) -> Self {
        self.visual_variant = variant.into();
        self
    }
}

// ---------------------------------------------------------------------------
// Enchantment
// ---------------------------------------------------------------------------

/// An enchantment that can be applied to a crafted item.
#[derive(Debug, Clone)]
pub struct Enchantment {
    /// Unique enchantment ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Description of the effect.
    pub description: String,
    /// Stat modifications: stat_name -> modifier_value.
    pub stat_modifiers: HashMap<String, f32>,
    /// Special effect ID (for gameplay logic).
    pub effect_id: Option<String>,
    /// Tier of the enchantment.
    pub tier: u32,
    /// Categories this enchantment can be applied to.
    pub applicable_categories: Vec<RecipeCategory>,
    /// Required enchanting skill level.
    pub skill_requirement: u32,
    /// Materials required to apply.
    pub materials: Vec<(String, u32)>,
    /// Success chance modifier.
    pub difficulty: f32,
}

impl Enchantment {
    /// Create a new enchantment.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            stat_modifiers: HashMap::new(),
            effect_id: None,
            tier: 1,
            applicable_categories: Vec::new(),
            skill_requirement: 0,
            materials: Vec::new(),
            difficulty: 1.0,
        }
    }

    /// Add a stat modifier.
    pub fn with_stat(mut self, stat: impl Into<String>, value: f32) -> Self {
        self.stat_modifiers.insert(stat.into(), value);
        self
    }

    /// Set a special effect.
    pub fn with_effect(mut self, effect_id: impl Into<String>) -> Self {
        self.effect_id = Some(effect_id.into());
        self
    }

    /// Set the tier.
    pub fn with_tier(mut self, tier: u32) -> Self {
        self.tier = tier;
        self
    }

    /// Set applicable categories.
    pub fn for_categories(mut self, categories: Vec<RecipeCategory>) -> Self {
        self.applicable_categories = categories;
        self
    }

    /// Check if this enchantment can be applied to a category.
    pub fn can_apply_to(&self, category: RecipeCategory) -> bool {
        self.applicable_categories.is_empty()
            || self.applicable_categories.contains(&category)
    }
}

// ---------------------------------------------------------------------------
// CraftedItem — the result of crafting
// ---------------------------------------------------------------------------

/// A crafted item with quality, stats, and enchantments.
#[derive(Debug, Clone)]
pub struct CraftedItem {
    /// Base item ID.
    pub item_id: String,
    /// Display name (may include quality prefix).
    pub name: String,
    /// Quality tier.
    pub quality: QualityTier,
    /// Final stats after all bonuses.
    pub stats: HashMap<String, f32>,
    /// Current durability.
    pub durability: f32,
    /// Maximum durability.
    pub max_durability: f32,
    /// Applied enchantments.
    pub enchantments: Vec<AppliedEnchantment>,
    /// Maximum enchantment slots.
    pub max_enchant_slots: usize,
    /// Material used (if applicable).
    pub material: Option<String>,
    /// Recipe that produced this item.
    pub recipe_id: String,
    /// Crafter entity ID.
    pub crafter: u64,
    /// Modification count.
    pub modifications: u32,
}

/// An enchantment applied to a specific item.
#[derive(Debug, Clone)]
pub struct AppliedEnchantment {
    /// Enchantment ID.
    pub enchantment_id: String,
    /// Enchantment name.
    pub name: String,
    /// Active stat modifiers.
    pub stat_modifiers: HashMap<String, f32>,
    /// Special effect ID.
    pub effect_id: Option<String>,
}

impl CraftedItem {
    /// Check if the item has room for more enchantments.
    pub fn can_enchant(&self) -> bool {
        self.enchantments.len() < self.max_enchant_slots
            && self.modifications < MAX_MODIFICATIONS as u32
    }

    /// Apply an enchantment. Returns true if successful.
    pub fn apply_enchantment(&mut self, enchantment: &Enchantment) -> bool {
        if !self.can_enchant() {
            return false;
        }

        // Check if already has this enchantment
        if self
            .enchantments
            .iter()
            .any(|e| e.enchantment_id == enchantment.id)
        {
            return false;
        }

        // Apply stat modifiers
        for (stat, &modifier) in &enchantment.stat_modifiers {
            *self.stats.entry(stat.clone()).or_insert(0.0) += modifier;
        }

        self.enchantments.push(AppliedEnchantment {
            enchantment_id: enchantment.id.clone(),
            name: enchantment.name.clone(),
            stat_modifiers: enchantment.stat_modifiers.clone(),
            effect_id: enchantment.effect_id.clone(),
        });

        self.modifications += 1;
        true
    }

    /// Remove an enchantment by ID.
    pub fn remove_enchantment(&mut self, enchantment_id: &str) -> bool {
        if let Some(idx) = self
            .enchantments
            .iter()
            .position(|e| e.enchantment_id == enchantment_id)
        {
            let removed = self.enchantments.remove(idx);
            // Reverse stat modifiers
            for (stat, modifier) in &removed.stat_modifiers {
                if let Some(val) = self.stats.get_mut(stat) {
                    *val -= modifier;
                }
            }
            true
        } else {
            false
        }
    }

    /// Get a stat value.
    pub fn get_stat(&self, stat: &str) -> f32 {
        self.stats.get(stat).copied().unwrap_or(0.0)
    }

    /// Get durability as a fraction (0..1).
    pub fn durability_fraction(&self) -> f32 {
        if self.max_durability <= 0.0 {
            return 0.0;
        }
        self.durability / self.max_durability
    }

    /// Degrade durability.
    pub fn degrade(&mut self, amount: f32) {
        self.durability = (self.durability - amount).max(0.0);
    }

    /// Repair durability.
    pub fn repair(&mut self, amount: f32) {
        self.durability = (self.durability + amount).min(self.max_durability);
    }

    /// Is the item broken?
    pub fn is_broken(&self) -> bool {
        self.durability <= 0.0
    }
}

// ---------------------------------------------------------------------------
// StatRoller — determines random stat values
// ---------------------------------------------------------------------------

/// Determines final stat values with randomness.
pub struct StatRoller {
    /// Seed for deterministic rolling.
    seed: u64,
}

impl StatRoller {
    /// Create a new stat roller with a seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Roll a stat value around a base with variance.
    pub fn roll_stat(&mut self, base: f32, variance: f32) -> f32 {
        let roll = self.next_f32();
        let deviation = (roll * 2.0 - 1.0) * variance;
        (base * (1.0 + deviation)).max(0.0)
    }

    /// Roll quality tier based on skill level and bonuses.
    pub fn roll_quality(
        &mut self,
        base_quality: QualityTier,
        max_quality: QualityTier,
        skill_level: u32,
        bonus: f32,
    ) -> QualityTier {
        let upgrade_chance =
            BASE_QUALITY_CHANCE + skill_level as f32 * SKILL_QUALITY_FACTOR + bonus;

        let mut quality = base_quality;
        loop {
            if quality >= max_quality {
                break;
            }
            let roll = self.next_f32();
            if roll < upgrade_chance {
                if let Some(next) = quality.upgrade() {
                    quality = next;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        quality
    }

    /// Roll an enchanting success check.
    pub fn roll_enchant_success(&mut self, skill_level: u32, difficulty: f32) -> bool {
        let chance = ENCHANT_BASE_CHANCE
            + skill_level as f32 * 0.005
            - (difficulty - 1.0) * 0.1;
        let roll = self.next_f32();
        roll < chance.clamp(0.05, 0.99)
    }

    /// Simple PRNG: xorshift64.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.seed = x;
        x
    }

    /// Generate a float in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 10000) as f32 / 10000.0
    }
}

// ---------------------------------------------------------------------------
// CraftingResult
// ---------------------------------------------------------------------------

/// Result of a crafting attempt.
#[derive(Debug)]
pub enum CraftingResult {
    /// Crafting succeeded, produced items.
    Success {
        items: Vec<CraftedItem>,
        xp_gained: f32,
        quality: QualityTier,
    },
    /// Missing ingredients.
    MissingIngredients(Vec<(String, u32)>),
    /// Missing crafting station.
    MissingStation(String),
    /// Insufficient skill level.
    InsufficientSkill {
        required: u32,
        current: u32,
    },
    /// Recipe not discovered.
    RecipeNotDiscovered,
    /// Recipe not found.
    RecipeNotFound,
    /// Station tier too low.
    StationTierTooLow {
        required: u32,
        current: u32,
    },
}

// ---------------------------------------------------------------------------
// RecipeBook — per-player recipe knowledge
// ---------------------------------------------------------------------------

/// Tracks which recipes a player has discovered and their crafting state.
pub struct RecipeBook {
    /// Owner entity ID.
    pub owner: u64,
    /// Discovered recipe IDs.
    discovered: HashSet<String>,
    /// Favorite recipes (for quick access).
    favorites: Vec<String>,
    /// Number of times each recipe has been crafted.
    craft_counts: HashMap<String, u32>,
    /// Skill levels by category.
    skill_levels: HashMap<RecipeCategory, u32>,
}

impl RecipeBook {
    /// Create a new recipe book.
    pub fn new(owner: u64) -> Self {
        Self {
            owner,
            discovered: HashSet::new(),
            favorites: Vec::new(),
            craft_counts: HashMap::new(),
            skill_levels: HashMap::new(),
        }
    }

    /// Discover a recipe.
    pub fn discover(&mut self, recipe_id: impl Into<String>) -> bool {
        self.discovered.insert(recipe_id.into())
    }

    /// Check if a recipe is discovered.
    pub fn is_discovered(&self, recipe_id: &str) -> bool {
        self.discovered.contains(recipe_id)
    }

    /// Get all discovered recipe IDs.
    pub fn discovered_recipes(&self) -> &HashSet<String> {
        &self.discovered
    }

    /// Toggle a recipe as favorite.
    pub fn toggle_favorite(&mut self, recipe_id: impl Into<String>) {
        let id = recipe_id.into();
        if let Some(pos) = self.favorites.iter().position(|f| f == &id) {
            self.favorites.remove(pos);
        } else {
            self.favorites.push(id);
        }
    }

    /// Get favorites.
    pub fn favorites(&self) -> &[String] {
        &self.favorites
    }

    /// Record a crafting completion.
    pub fn record_craft(&mut self, recipe_id: &str, category: RecipeCategory) {
        *self.craft_counts.entry(recipe_id.to_string()).or_insert(0) += 1;
        // Auto-level skills
        let count = *self
            .craft_counts
            .get(recipe_id)
            .unwrap_or(&0);
        if count % 5 == 0 {
            *self.skill_levels.entry(category).or_insert(0) += 1;
        }
    }

    /// Get skill level for a category.
    pub fn skill_level(&self, category: RecipeCategory) -> u32 {
        self.skill_levels.get(&category).copied().unwrap_or(0)
    }

    /// Set skill level directly.
    pub fn set_skill_level(&mut self, category: RecipeCategory, level: u32) {
        self.skill_levels.insert(category, level);
    }

    /// Get craft count for a recipe.
    pub fn craft_count(&self, recipe_id: &str) -> u32 {
        self.craft_counts.get(recipe_id).copied().unwrap_or(0)
    }

    /// Total crafts across all recipes.
    pub fn total_crafts(&self) -> u32 {
        self.craft_counts.values().sum()
    }
}

// ---------------------------------------------------------------------------
// CraftingSystem
// ---------------------------------------------------------------------------

/// Top-level crafting system.
pub struct CraftingSystem {
    /// All available recipes.
    recipes: HashMap<String, Recipe>,
    /// Registered crafting stations.
    stations: HashMap<String, CraftingStation>,
    /// Material bonuses.
    material_bonuses: HashMap<String, MaterialBonus>,
    /// Available enchantments.
    enchantments: HashMap<String, Enchantment>,
    /// Per-player recipe books.
    recipe_books: HashMap<u64, RecipeBook>,
    /// Stat roller.
    roller: StatRoller,
}

impl CraftingSystem {
    /// Create a new crafting system.
    pub fn new(seed: u64) -> Self {
        Self {
            recipes: HashMap::new(),
            stations: HashMap::new(),
            material_bonuses: HashMap::new(),
            enchantments: HashMap::new(),
            recipe_books: HashMap::new(),
            roller: StatRoller::new(seed),
        }
    }

    /// Register a recipe.
    pub fn register_recipe(&mut self, recipe: Recipe) {
        self.recipes.insert(recipe.id.clone(), recipe);
    }

    /// Register a crafting station type.
    pub fn register_station(&mut self, station: CraftingStation) {
        self.stations.insert(station.id.clone(), station);
    }

    /// Register a material bonus.
    pub fn register_material_bonus(&mut self, bonus: MaterialBonus) {
        self.material_bonuses.insert(bonus.material_id.clone(), bonus);
    }

    /// Register an enchantment.
    pub fn register_enchantment(&mut self, enchantment: Enchantment) {
        self.enchantments
            .insert(enchantment.id.clone(), enchantment);
    }

    /// Get or create a player's recipe book.
    pub fn get_recipe_book(&mut self, player: u64) -> &mut RecipeBook {
        self.recipe_books
            .entry(player)
            .or_insert_with(|| RecipeBook::new(player))
    }

    /// Get a recipe by ID.
    pub fn get_recipe(&self, recipe_id: &str) -> Option<&Recipe> {
        self.recipes.get(recipe_id)
    }

    /// Get recipes by category.
    pub fn recipes_by_category(&self, category: RecipeCategory) -> Vec<&Recipe> {
        self.recipes
            .values()
            .filter(|r| r.category == category)
            .collect()
    }

    /// Get all recipe categories that have at least one recipe.
    pub fn available_categories(&self) -> Vec<RecipeCategory> {
        let mut cats: HashSet<RecipeCategory> = HashSet::new();
        for recipe in self.recipes.values() {
            cats.insert(recipe.category);
        }
        cats.into_iter().collect()
    }

    /// Attempt to craft a recipe.
    pub fn craft(
        &mut self,
        player: u64,
        recipe_id: &str,
        available_items: &HashMap<String, u32>,
        station_id: Option<&str>,
        primary_material: Option<&str>,
    ) -> CraftingResult {
        // Look up recipe
        let recipe = match self.recipes.get(recipe_id) {
            Some(r) => r.clone(),
            None => return CraftingResult::RecipeNotFound,
        };

        // Check discovery
        if recipe.requires_discovery {
            let book = self
                .recipe_books
                .entry(player)
                .or_insert_with(|| RecipeBook::new(player));
            if !book.is_discovered(recipe_id) {
                return CraftingResult::RecipeNotDiscovered;
            }
        }

        // Check ingredients
        if !recipe.can_craft(available_items) {
            return CraftingResult::MissingIngredients(
                recipe.missing_ingredients(available_items),
            );
        }

        // Check station
        let station_bonus = if let Some(ref required) = recipe.required_station {
            match station_id {
                Some(sid) => {
                    if sid != required {
                        return CraftingResult::MissingStation(required.clone());
                    }
                    if let Some(station) = self.stations.get(sid) {
                        if station.tier < recipe.min_station_tier {
                            return CraftingResult::StationTierTooLow {
                                required: recipe.min_station_tier,
                                current: station.tier,
                            };
                        }
                        station.quality_bonus
                    } else {
                        return CraftingResult::MissingStation(required.clone());
                    }
                }
                None => return CraftingResult::MissingStation(required.clone()),
            }
        } else {
            0.0
        };

        // Check skill
        let book = self
            .recipe_books
            .entry(player)
            .or_insert_with(|| RecipeBook::new(player));
        let skill_level = book.skill_level(recipe.category);

        if let Some((_, required_level)) = &recipe.skill_requirement {
            if skill_level < *required_level {
                return CraftingResult::InsufficientSkill {
                    required: *required_level,
                    current: skill_level,
                };
            }
        }

        // Roll quality
        let quality = self.roller.roll_quality(
            recipe.min_quality,
            recipe.max_quality,
            skill_level,
            station_bonus,
        );

        // Get material bonus
        let mat_bonus = primary_material
            .and_then(|m| self.material_bonuses.get(m));

        // Produce outputs
        let mut items = Vec::new();
        for output in &recipe.outputs {
            let roll = self.roller.next_f32();
            if roll >= output.chance {
                continue;
            }

            // Calculate stats
            let mut stats = HashMap::new();
            for (stat, &base) in &output.base_stats {
                let mut value = self.roller.roll_stat(base, MAX_STAT_VARIANCE);
                value *= quality.stat_multiplier();

                // Material bonus
                if let Some(bonus) = mat_bonus {
                    if let Some(&mat_bonus_val) = bonus.stat_bonuses.get(stat) {
                        value += mat_bonus_val;
                    }
                }

                stats.insert(stat.clone(), value);
            }

            // Calculate durability
            let mut durability = output.base_durability * quality.durability_multiplier();
            if let Some(bonus) = mat_bonus {
                durability *= bonus.durability_multiplier;
            }

            let quality_prefix = if quality != QualityTier::Common {
                format!("{} ", quality.name())
            } else {
                String::new()
            };

            let item = CraftedItem {
                item_id: output.item_id.clone(),
                name: format!("{}{}", quality_prefix, output.item_id),
                quality,
                stats,
                durability,
                max_durability: durability,
                enchantments: Vec::new(),
                max_enchant_slots: quality.enchant_slots(),
                material: primary_material.map(|s| s.to_string()),
                recipe_id: recipe_id.to_string(),
                crafter: player,
                modifications: 0,
            };

            items.push(item);
        }

        // Record craft
        let book = self.recipe_books.get_mut(&player).unwrap();
        book.record_craft(recipe_id, recipe.category);

        CraftingResult::Success {
            items,
            xp_gained: recipe.xp_reward,
            quality,
        }
    }

    /// Apply an enchantment to a crafted item.
    pub fn enchant(
        &mut self,
        player: u64,
        item: &mut CraftedItem,
        enchantment_id: &str,
    ) -> EnchantResult {
        let enchantment = match self.enchantments.get(enchantment_id) {
            Some(e) => e.clone(),
            None => return EnchantResult::EnchantmentNotFound,
        };

        if !item.can_enchant() {
            return EnchantResult::NoSlots;
        }

        let book = self
            .recipe_books
            .entry(player)
            .or_insert_with(|| RecipeBook::new(player));
        let skill = book.skill_level(RecipeCategory::Enchanting);

        if skill < enchantment.skill_requirement {
            return EnchantResult::InsufficientSkill;
        }

        let success = self
            .roller
            .roll_enchant_success(skill, enchantment.difficulty);

        if success {
            item.apply_enchantment(&enchantment);
            EnchantResult::Success
        } else {
            // Failure: lose durability
            let penalty = item.max_durability * ENCHANT_FAILURE_PENALTY;
            item.degrade(penalty);
            EnchantResult::Failed { durability_lost: penalty }
        }
    }

    /// Get recipe count.
    pub fn recipe_count(&self) -> usize {
        self.recipes.len()
    }

    /// Get station count.
    pub fn station_count(&self) -> usize {
        self.stations.len()
    }

    /// Get enchantment count.
    pub fn enchantment_count(&self) -> usize {
        self.enchantments.len()
    }
}

/// Result of an enchanting attempt.
#[derive(Debug)]
pub enum EnchantResult {
    /// Enchantment applied successfully.
    Success,
    /// No available enchantment slots.
    NoSlots,
    /// Enchantment not found.
    EnchantmentNotFound,
    /// Insufficient enchanting skill.
    InsufficientSkill,
    /// Enchantment failed.
    Failed { durability_lost: f32 },
}

impl Default for CraftingSystem {
    fn default() -> Self {
        Self::new(42)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_tiers() {
        assert!(QualityTier::Legendary.stat_multiplier() > QualityTier::Common.stat_multiplier());
        assert_eq!(QualityTier::Common.upgrade(), Some(QualityTier::Uncommon));
        assert_eq!(QualityTier::Legendary.upgrade(), None);
    }

    #[test]
    fn test_recipe_ingredients() {
        let recipe = Recipe::new("iron_sword", "Iron Sword", RecipeCategory::Weaponsmithing)
            .ingredient(Ingredient::new("iron_ingot", 3))
            .ingredient(Ingredient::new("leather_strip", 1))
            .output(CraftingOutput::new("iron_sword", 1).with_stat("damage", 25.0));

        let mut available = HashMap::new();
        available.insert("iron_ingot".to_string(), 5);
        available.insert("leather_strip".to_string(), 2);

        assert!(recipe.can_craft(&available));

        available.insert("iron_ingot".to_string(), 1);
        assert!(!recipe.can_craft(&available));
    }

    #[test]
    fn test_stat_roller() {
        let mut roller = StatRoller::new(12345);
        let value = roller.roll_stat(100.0, 0.2);
        assert!(value >= 80.0 && value <= 120.0);
    }

    #[test]
    fn test_crafting_system() {
        let mut system = CraftingSystem::new(42);

        system.register_recipe(
            Recipe::new("basic_potion", "Health Potion", RecipeCategory::Alchemy)
                .ingredient(Ingredient::new("herb", 2))
                .ingredient(Ingredient::new("water", 1))
                .output(CraftingOutput::new("health_potion", 1).with_stat("heal_amount", 50.0)),
        );

        let mut available = HashMap::new();
        available.insert("herb".to_string(), 5);
        available.insert("water".to_string(), 3);

        let result = system.craft(1, "basic_potion", &available, None, None);
        match result {
            CraftingResult::Success { items, .. } => {
                assert!(!items.is_empty());
            }
            other => panic!("Expected success, got {:?}", other),
        }
    }

    #[test]
    fn test_enchantment() {
        let mut item = CraftedItem {
            item_id: "sword".into(),
            name: "Iron Sword".into(),
            quality: QualityTier::Rare,
            stats: HashMap::from([("damage".to_string(), 25.0)]),
            durability: 100.0,
            max_durability: 100.0,
            enchantments: Vec::new(),
            max_enchant_slots: 2,
            material: None,
            recipe_id: "iron_sword".into(),
            crafter: 1,
            modifications: 0,
        };

        let enchant = Enchantment::new("fire_damage", "Flame")
            .with_stat("fire_damage", 10.0);

        assert!(item.apply_enchantment(&enchant));
        assert_eq!(item.get_stat("fire_damage"), 10.0);
        assert_eq!(item.enchantments.len(), 1);
    }

    #[test]
    fn test_recipe_book() {
        let mut book = RecipeBook::new(1);
        assert!(!book.is_discovered("sword_recipe"));

        book.discover("sword_recipe");
        assert!(book.is_discovered("sword_recipe"));

        book.record_craft("sword_recipe", RecipeCategory::Weaponsmithing);
        assert_eq!(book.craft_count("sword_recipe"), 1);
    }
}
