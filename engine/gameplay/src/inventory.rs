//! Inventory, item management, loot tables, and crafting.
//!
//! Provides a complete inventory system suitable for RPGs, survival games,
//! and action games. Features include:
//!
//! - Fixed-slot inventories with add/remove/swap/split/merge operations
//! - Item stacking with configurable max stack sizes
//! - Weight limits and encumbrance
//! - Equipment slots (head, chest, legs, hands, etc.)
//! - Weighted loot tables with rarity tiers
//! - Crafting recipes with input/output validation
//!
//! # Data model
//!
//! Items are identified by a string `item_id` and described by [`ItemDefinition`].
//! An [`ItemStack`] is a runtime instance of one or more items of the same type.
//! [`Inventory`] holds a fixed array of optional slots, each containing at most
//! one [`ItemStack`].

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Item rarity
// ---------------------------------------------------------------------------

/// Rarity tier for items and loot drops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum ItemRarity {
    /// White / junk tier.
    Common,
    /// Green tier.
    Uncommon,
    /// Blue tier.
    Rare,
    /// Purple tier.
    Epic,
    /// Orange / gold tier.
    Legendary,
    /// Red / unique tier.
    Mythic,
}

impl ItemRarity {
    /// Base drop weight multiplier (lower = rarer).
    pub fn base_weight(&self) -> f32 {
        match self {
            Self::Common => 100.0,
            Self::Uncommon => 50.0,
            Self::Rare => 20.0,
            Self::Epic => 5.0,
            Self::Legendary => 1.0,
            Self::Mythic => 0.2,
        }
    }

    /// Color hex string for UI display.
    pub fn color_hex(&self) -> &'static str {
        match self {
            Self::Common => "#FFFFFF",
            Self::Uncommon => "#1EFF00",
            Self::Rare => "#0070DD",
            Self::Epic => "#A335EE",
            Self::Legendary => "#FF8000",
            Self::Mythic => "#FF0000",
        }
    }
}

impl Default for ItemRarity {
    fn default() -> Self {
        Self::Common
    }
}

// ---------------------------------------------------------------------------
// Item category
// ---------------------------------------------------------------------------

/// Broad category for items, used for filtering and sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ItemCategory {
    Weapon,
    Armor,
    Consumable,
    Material,
    QuestItem,
    Currency,
    Ammo,
    Misc,
}

impl Default for ItemCategory {
    fn default() -> Self {
        Self::Misc
    }
}

// ---------------------------------------------------------------------------
// Equipment slot
// ---------------------------------------------------------------------------

/// Body slots for equipping items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EquipmentSlot {
    Head,
    Chest,
    Legs,
    Feet,
    Hands,
    Shoulders,
    Back,
    Neck,
    RingLeft,
    RingRight,
    MainHand,
    OffHand,
    TwoHand,
    Ranged,
    Ammo,
}

impl EquipmentSlot {
    /// All equipment slots as a static array.
    pub const ALL: &'static [EquipmentSlot] = &[
        Self::Head,
        Self::Chest,
        Self::Legs,
        Self::Feet,
        Self::Hands,
        Self::Shoulders,
        Self::Back,
        Self::Neck,
        Self::RingLeft,
        Self::RingRight,
        Self::MainHand,
        Self::OffHand,
        Self::TwoHand,
        Self::Ranged,
        Self::Ammo,
    ];
}

// ---------------------------------------------------------------------------
// Item definition (static data)
// ---------------------------------------------------------------------------

/// Static definition of an item type, loaded from data files.
///
/// This describes the *template* for items. Runtime instances are
/// [`ItemStack`]s referencing a definition by `item_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemDefinition {
    /// Unique identifier for this item type (e.g., "iron_sword").
    pub item_id: String,
    /// Display name.
    pub display_name: String,
    /// Description / flavor text.
    pub description: String,
    /// Category.
    pub category: ItemCategory,
    /// Rarity.
    pub rarity: ItemRarity,
    /// Maximum stack size (1 = not stackable).
    pub max_stack: u32,
    /// Weight per unit.
    pub weight: f32,
    /// Base value (for selling / buying).
    pub base_value: u32,
    /// Icon asset path.
    pub icon: String,
    /// Equipment slot this item can be equipped in (if any).
    pub equip_slot: Option<EquipmentSlot>,
    /// Whether this item is consumed on use.
    pub consumable: bool,
    /// Whether this item can be dropped.
    pub droppable: bool,
    /// Whether this item can be traded.
    pub tradeable: bool,
    /// Custom properties (damage, armor, effects, etc.).
    pub properties: HashMap<String, ItemProperty>,
}

impl Default for ItemDefinition {
    fn default() -> Self {
        Self {
            item_id: String::new(),
            display_name: String::new(),
            description: String::new(),
            category: ItemCategory::Misc,
            rarity: ItemRarity::Common,
            max_stack: 1,
            weight: 0.1,
            base_value: 0,
            icon: String::new(),
            equip_slot: None,
            consumable: false,
            droppable: true,
            tradeable: true,
            properties: HashMap::new(),
        }
    }
}

/// A typed property value attached to an item definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemProperty {
    Int(i32),
    Float(f32),
    String(String),
    Bool(bool),
}

// ---------------------------------------------------------------------------
// Item registry
// ---------------------------------------------------------------------------

/// Registry of all item definitions. Typically loaded once at startup.
#[derive(Debug, Default)]
pub struct ItemRegistry {
    definitions: HashMap<String, ItemDefinition>,
}

impl ItemRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
        }
    }

    /// Register an item definition.
    pub fn register(&mut self, definition: ItemDefinition) {
        self.definitions
            .insert(definition.item_id.clone(), definition);
    }

    /// Look up a definition by item id.
    pub fn get(&self, item_id: &str) -> Option<&ItemDefinition> {
        self.definitions.get(item_id)
    }

    /// Check if an item id is registered.
    pub fn contains(&self, item_id: &str) -> bool {
        self.definitions.contains_key(item_id)
    }

    /// Number of registered definitions.
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    /// Iterate over all definitions.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ItemDefinition)> {
        self.definitions.iter()
    }
}

// ---------------------------------------------------------------------------
// Item stack (runtime instance)
// ---------------------------------------------------------------------------

/// A stack of one or more items of the same type.
///
/// This is the runtime representation stored in inventory slots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemStack {
    /// The item type id (references an [`ItemDefinition`]).
    pub item_id: String,
    /// Number of items in this stack (>= 1).
    pub quantity: u32,
    /// Instance-specific data (e.g., durability, enchantments).
    pub instance_data: Option<ItemInstanceData>,
}

/// Per-instance data for items that have mutable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemInstanceData {
    /// Current durability (if applicable).
    pub durability: Option<f32>,
    /// Maximum durability.
    pub max_durability: Option<f32>,
    /// Custom instance properties (e.g., random stats, sockets).
    pub properties: HashMap<String, ItemProperty>,
}

impl ItemStack {
    /// Create a new item stack.
    pub fn new(item_id: impl Into<String>, quantity: u32) -> Self {
        Self {
            item_id: item_id.into(),
            quantity: quantity.max(1),
            instance_data: None,
        }
    }

    /// Create a stack with instance data.
    pub fn with_instance_data(
        item_id: impl Into<String>,
        quantity: u32,
        data: ItemInstanceData,
    ) -> Self {
        Self {
            item_id: item_id.into(),
            quantity: quantity.max(1),
            instance_data: Some(data),
        }
    }

    /// Total weight of this stack, using the registry for per-unit weight.
    pub fn total_weight(&self, registry: &ItemRegistry) -> f32 {
        registry
            .get(&self.item_id)
            .map(|def| def.weight * self.quantity as f32)
            .unwrap_or(0.0)
    }

    /// Total value of this stack.
    pub fn total_value(&self, registry: &ItemRegistry) -> u32 {
        registry
            .get(&self.item_id)
            .map(|def| def.base_value * self.quantity)
            .unwrap_or(0)
    }

    /// Whether this stack can merge with another (same item, both below max).
    pub fn can_merge_with(&self, other: &Self, registry: &ItemRegistry) -> bool {
        if self.item_id != other.item_id {
            return false;
        }
        // Items with instance data generally can't stack.
        if self.instance_data.is_some() || other.instance_data.is_some() {
            return false;
        }
        let max_stack = registry
            .get(&self.item_id)
            .map(|d| d.max_stack)
            .unwrap_or(1);
        max_stack > 1 && self.quantity < max_stack
    }

    /// Try to merge another stack into this one. Returns leftover quantity
    /// that did not fit.
    pub fn merge_from(&mut self, other: &mut Self, registry: &ItemRegistry) -> u32 {
        if self.item_id != other.item_id {
            return other.quantity;
        }
        let max_stack = registry
            .get(&self.item_id)
            .map(|d| d.max_stack)
            .unwrap_or(1);
        let space = max_stack.saturating_sub(self.quantity);
        let transfer = space.min(other.quantity);
        self.quantity += transfer;
        other.quantity -= transfer;
        other.quantity
    }

    /// Split this stack, removing `count` items and returning them as a new
    /// stack. Returns `None` if count >= current quantity (can't split all).
    pub fn split(&mut self, count: u32) -> Option<Self> {
        if count == 0 || count >= self.quantity {
            return None;
        }
        self.quantity -= count;
        Some(Self {
            item_id: self.item_id.clone(),
            quantity: count,
            instance_data: None, // Splits don't carry instance data.
        })
    }

    /// Whether this stack is empty (quantity == 0).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.quantity == 0
    }

    /// Current durability as a fraction (0..1), if applicable.
    pub fn durability_fraction(&self) -> Option<f32> {
        self.instance_data.as_ref().and_then(|d| {
            match (d.durability, d.max_durability) {
                (Some(cur), Some(max)) if max > 0.0 => Some(cur / max),
                _ => None,
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Inventory result
// ---------------------------------------------------------------------------

/// Errors that can occur during inventory operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum InventoryError {
    #[error("inventory is full")]
    Full,
    #[error("slot index {0} is out of range")]
    SlotOutOfRange(usize),
    #[error("slot {0} is empty")]
    SlotEmpty(usize),
    #[error("slot {0} is occupied")]
    SlotOccupied(usize),
    #[error("weight limit exceeded (current: {current:.1}, adding: {adding:.1}, limit: {limit:.1})")]
    WeightLimitExceeded {
        current: f32,
        adding: f32,
        limit: f32,
    },
    #[error("item '{0}' not found in registry")]
    UnknownItem(String),
    #[error("cannot stack items with instance data")]
    CannotStack,
    #[error("item '{0}' cannot be equipped in slot {1:?}")]
    WrongEquipSlot(String, EquipmentSlot),
}

/// Result type for inventory operations.
pub type InventoryResult<T> = Result<T, InventoryError>;

// ---------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------

/// A fixed-size inventory with optional weight limit.
///
/// Slots are indexed from `0` to `capacity - 1`. Each slot holds an optional
/// [`ItemStack`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inventory {
    /// Fixed array of inventory slots.
    slots: Vec<Option<ItemStack>>,
    /// Maximum total weight (0 or negative = unlimited).
    weight_limit: f32,
    /// Display name for this inventory (e.g., "Backpack", "Chest").
    pub name: String,
}

impl Inventory {
    /// Create a new inventory with the given number of slots.
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: vec![None; capacity],
            weight_limit: 0.0,
            name: String::from("Inventory"),
        }
    }

    /// Create an inventory with a weight limit.
    pub fn with_weight_limit(capacity: usize, weight_limit: f32) -> Self {
        Self {
            slots: vec![None; capacity],
            weight_limit,
            name: String::from("Inventory"),
        }
    }

    /// Number of total slots.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Number of occupied slots.
    pub fn used_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Number of empty slots.
    pub fn free_slots(&self) -> usize {
        self.capacity() - self.used_slots()
    }

    /// Whether the inventory is completely full.
    pub fn is_full(&self) -> bool {
        self.free_slots() == 0
    }

    /// Get an immutable reference to a slot.
    pub fn get_slot(&self, index: usize) -> InventoryResult<Option<&ItemStack>> {
        self.slots
            .get(index)
            .map(|s| s.as_ref())
            .ok_or(InventoryError::SlotOutOfRange(index))
    }

    /// Get a mutable reference to a slot.
    pub fn get_slot_mut(&mut self, index: usize) -> InventoryResult<&mut Option<ItemStack>> {
        if index >= self.slots.len() {
            return Err(InventoryError::SlotOutOfRange(index));
        }
        Ok(&mut self.slots[index])
    }

    /// Total weight of all items in the inventory.
    pub fn total_weight(&self, registry: &ItemRegistry) -> f32 {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|stack| stack.total_weight(registry))
            .sum()
    }

    /// Remaining weight capacity.
    pub fn remaining_weight(&self, registry: &ItemRegistry) -> f32 {
        if self.weight_limit <= 0.0 {
            return f32::MAX;
        }
        (self.weight_limit - self.total_weight(registry)).max(0.0)
    }

    /// Whether adding an item would exceed the weight limit.
    fn would_exceed_weight(&self, item_id: &str, quantity: u32, registry: &ItemRegistry) -> bool {
        if self.weight_limit <= 0.0 {
            return false;
        }
        let unit_weight = registry.get(item_id).map(|d| d.weight).unwrap_or(0.0);
        let adding = unit_weight * quantity as f32;
        self.total_weight(registry) + adding > self.weight_limit
    }

    /// Add items to the inventory, stacking with existing compatible stacks
    /// first, then filling empty slots. Returns the number of items that
    /// could not be added.
    pub fn add_item(
        &mut self,
        item_id: &str,
        mut quantity: u32,
        registry: &ItemRegistry,
    ) -> InventoryResult<u32> {
        if quantity == 0 {
            return Ok(0);
        }

        // Weight check.
        if self.would_exceed_weight(item_id, quantity, registry) {
            let current = self.total_weight(registry);
            let unit_w = registry.get(item_id).map(|d| d.weight).unwrap_or(0.0);
            return Err(InventoryError::WeightLimitExceeded {
                current,
                adding: unit_w * quantity as f32,
                limit: self.weight_limit,
            });
        }

        let max_stack = registry
            .get(item_id)
            .map(|d| d.max_stack)
            .unwrap_or(1);

        // Phase 1: try to merge into existing stacks.
        for slot in self.slots.iter_mut() {
            if quantity == 0 {
                break;
            }
            if let Some(stack) = slot {
                if stack.item_id == item_id
                    && stack.instance_data.is_none()
                    && max_stack > 1
                {
                    let space = max_stack.saturating_sub(stack.quantity);
                    let transfer = space.min(quantity);
                    stack.quantity += transfer;
                    quantity -= transfer;
                }
            }
        }

        // Phase 2: fill empty slots.
        for slot in self.slots.iter_mut() {
            if quantity == 0 {
                break;
            }
            if slot.is_none() {
                let stack_qty = quantity.min(max_stack);
                *slot = Some(ItemStack::new(item_id, stack_qty));
                quantity -= stack_qty;
            }
        }

        Ok(quantity)
    }

    /// Add an item stack directly to a specific slot.
    pub fn add_to_slot(
        &mut self,
        index: usize,
        stack: ItemStack,
        registry: &ItemRegistry,
    ) -> InventoryResult<Option<ItemStack>> {
        if index >= self.slots.len() {
            return Err(InventoryError::SlotOutOfRange(index));
        }

        if self.would_exceed_weight(&stack.item_id, stack.quantity, registry) {
            let current = self.total_weight(registry);
            let unit_w = registry.get(&stack.item_id).map(|d| d.weight).unwrap_or(0.0);
            return Err(InventoryError::WeightLimitExceeded {
                current,
                adding: unit_w * stack.quantity as f32,
                limit: self.weight_limit,
            });
        }

        // If slot is occupied, try to merge.
        if let Some(existing) = &mut self.slots[index] {
            if existing.item_id == stack.item_id && existing.instance_data.is_none() {
                let max_stack = registry
                    .get(&stack.item_id)
                    .map(|d| d.max_stack)
                    .unwrap_or(1);
                let space = max_stack.saturating_sub(existing.quantity);
                let transfer = space.min(stack.quantity);
                existing.quantity += transfer;
                let leftover = stack.quantity - transfer;
                if leftover > 0 {
                    return Ok(Some(ItemStack::new(&stack.item_id, leftover)));
                }
                return Ok(None);
            }
            // Different item -- return the incoming stack.
            return Err(InventoryError::SlotOccupied(index));
        }

        self.slots[index] = Some(stack);
        Ok(None)
    }

    /// Remove a quantity of an item by id. Returns actual number removed.
    pub fn remove_item(&mut self, item_id: &str, mut quantity: u32) -> u32 {
        let mut removed = 0u32;
        for slot in self.slots.iter_mut() {
            if quantity == 0 {
                break;
            }
            if let Some(stack) = slot {
                if stack.item_id == item_id {
                    let take = stack.quantity.min(quantity);
                    stack.quantity -= take;
                    quantity -= take;
                    removed += take;
                    if stack.quantity == 0 {
                        *slot = None;
                    }
                }
            }
        }
        removed
    }

    /// Remove the entire stack from a slot and return it.
    pub fn take_slot(&mut self, index: usize) -> InventoryResult<ItemStack> {
        if index >= self.slots.len() {
            return Err(InventoryError::SlotOutOfRange(index));
        }
        self.slots[index]
            .take()
            .ok_or(InventoryError::SlotEmpty(index))
    }

    /// Swap the contents of two slots.
    pub fn swap_slots(&mut self, a: usize, b: usize) -> InventoryResult<()> {
        if a >= self.slots.len() {
            return Err(InventoryError::SlotOutOfRange(a));
        }
        if b >= self.slots.len() {
            return Err(InventoryError::SlotOutOfRange(b));
        }
        self.slots.swap(a, b);
        Ok(())
    }

    /// Split a stack at the given slot, putting `count` items into `target_slot`.
    pub fn split_stack(
        &mut self,
        source_slot: usize,
        target_slot: usize,
        count: u32,
    ) -> InventoryResult<()> {
        if source_slot >= self.slots.len() {
            return Err(InventoryError::SlotOutOfRange(source_slot));
        }
        if target_slot >= self.slots.len() {
            return Err(InventoryError::SlotOutOfRange(target_slot));
        }
        if self.slots[target_slot].is_some() {
            return Err(InventoryError::SlotOccupied(target_slot));
        }

        let split_stack = self.slots[source_slot]
            .as_mut()
            .ok_or(InventoryError::SlotEmpty(source_slot))?
            .split(count);

        match split_stack {
            Some(new_stack) => {
                self.slots[target_slot] = Some(new_stack);
                Ok(())
            }
            None => Ok(()), // Split of 0 or all -- no-op.
        }
    }

    /// Merge the stack at `source_slot` into `target_slot`.
    pub fn merge_slots(
        &mut self,
        source_slot: usize,
        target_slot: usize,
        registry: &ItemRegistry,
    ) -> InventoryResult<()> {
        if source_slot >= self.slots.len() {
            return Err(InventoryError::SlotOutOfRange(source_slot));
        }
        if target_slot >= self.slots.len() {
            return Err(InventoryError::SlotOutOfRange(target_slot));
        }
        if source_slot == target_slot {
            return Ok(());
        }

        // Take source out temporarily.
        let mut source = self.slots[source_slot]
            .take()
            .ok_or(InventoryError::SlotEmpty(source_slot))?;

        if let Some(target) = &mut self.slots[target_slot] {
            let leftover = target.merge_from(&mut source, registry);
            if leftover > 0 {
                // Put leftover back in the source slot.
                self.slots[source_slot] = Some(source);
            }
        } else {
            // Target is empty -- just move the stack.
            self.slots[target_slot] = Some(source);
        }

        Ok(())
    }

    /// Count total quantity of a specific item across all slots.
    pub fn count_item(&self, item_id: &str) -> u32 {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.item_id == item_id)
            .map(|s| s.quantity)
            .sum()
    }

    /// Whether the inventory contains at least `quantity` of the given item.
    pub fn has_item(&self, item_id: &str, quantity: u32) -> bool {
        self.count_item(item_id) >= quantity
    }

    /// Find the first slot containing the given item.
    pub fn find_item(&self, item_id: &str) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| s.as_ref().is_some_and(|stack| stack.item_id == item_id))
    }

    /// Find the first empty slot.
    pub fn find_empty_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| s.is_none())
    }

    /// Sort inventory: group by item_id, consolidate stacks.
    pub fn sort(&mut self, registry: &ItemRegistry) {
        // Collect all items.
        let mut items: Vec<ItemStack> = self
            .slots
            .iter_mut()
            .filter_map(|s| s.take())
            .collect();

        // Sort by category, rarity, then name.
        items.sort_by(|a, b| {
            let def_a = registry.get(&a.item_id);
            let def_b = registry.get(&b.item_id);
            let cat_a = def_a.map(|d| d.category as u8).unwrap_or(255);
            let cat_b = def_b.map(|d| d.category as u8).unwrap_or(255);
            cat_a
                .cmp(&cat_b)
                .then_with(|| {
                    let rar_a = def_a.map(|d| d.rarity).unwrap_or(ItemRarity::Common);
                    let rar_b = def_b.map(|d| d.rarity).unwrap_or(ItemRarity::Common);
                    rar_b.cmp(&rar_a) // Higher rarity first.
                })
                .then_with(|| a.item_id.cmp(&b.item_id))
        });

        // Consolidate stacks.
        let mut consolidated: Vec<ItemStack> = Vec::new();
        for item in items {
            let max_stack = registry
                .get(&item.item_id)
                .map(|d| d.max_stack)
                .unwrap_or(1);

            // Try to merge with last consolidated stack.
            let merged = if let Some(last) = consolidated.last_mut() {
                if last.item_id == item.item_id
                    && last.instance_data.is_none()
                    && item.instance_data.is_none()
                    && last.quantity < max_stack
                {
                    let space = max_stack - last.quantity;
                    let transfer = space.min(item.quantity);
                    last.quantity += transfer;
                    let leftover = item.quantity - transfer;
                    if leftover > 0 {
                        consolidated.push(ItemStack::new(&item.item_id, leftover));
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if !merged {
                consolidated.push(item);
            }
        }

        // Place back into slots.
        for (i, slot) in self.slots.iter_mut().enumerate() {
            *slot = consolidated.get(i).cloned();
        }
    }

    /// Clear all slots.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
    }

    /// Iterate over all occupied slots with their indices.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &ItemStack)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|stack| (i, stack)))
    }
}

impl genovo_ecs::Component for Inventory {}

// ---------------------------------------------------------------------------
// Equipment
// ---------------------------------------------------------------------------

/// Equipped item slots for a character.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Equipment {
    /// Map of slot -> equipped item stack.
    slots: HashMap<EquipmentSlot, ItemStack>,
}

impl Equipment {
    /// Create empty equipment.
    pub fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }

    /// Equip an item in the appropriate slot. Returns the previously equipped
    /// item, if any.
    pub fn equip(
        &mut self,
        stack: ItemStack,
        slot: EquipmentSlot,
        registry: &ItemRegistry,
    ) -> InventoryResult<Option<ItemStack>> {
        // Validate that this item can go in this slot.
        if let Some(def) = registry.get(&stack.item_id) {
            if let Some(valid_slot) = def.equip_slot {
                if valid_slot != slot {
                    return Err(InventoryError::WrongEquipSlot(
                        stack.item_id.clone(),
                        slot,
                    ));
                }
            }
        }

        let prev = self.slots.insert(slot, stack);
        Ok(prev)
    }

    /// Unequip the item in the given slot.
    pub fn unequip(&mut self, slot: EquipmentSlot) -> Option<ItemStack> {
        self.slots.remove(&slot)
    }

    /// Get the equipped item in a slot.
    pub fn get(&self, slot: EquipmentSlot) -> Option<&ItemStack> {
        self.slots.get(&slot)
    }

    /// Whether a slot is occupied.
    pub fn is_equipped(&self, slot: EquipmentSlot) -> bool {
        self.slots.contains_key(&slot)
    }

    /// Iterate over all equipped items.
    pub fn iter(&self) -> impl Iterator<Item = (&EquipmentSlot, &ItemStack)> {
        self.slots.iter()
    }

    /// Number of equipped items.
    pub fn count(&self) -> usize {
        self.slots.len()
    }

    /// Total weight of equipped items.
    pub fn total_weight(&self, registry: &ItemRegistry) -> f32 {
        self.slots
            .values()
            .map(|s| s.total_weight(registry))
            .sum()
    }
}

impl genovo_ecs::Component for Equipment {}

// ---------------------------------------------------------------------------
// Loot table
// ---------------------------------------------------------------------------

/// A single entry in a loot table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LootEntry {
    /// Item id to drop.
    pub item_id: String,
    /// Drop weight (higher = more likely).
    pub weight: f32,
    /// Minimum quantity (inclusive).
    pub min_quantity: u32,
    /// Maximum quantity (inclusive).
    pub max_quantity: u32,
    /// Minimum rarity required for this entry to be considered.
    pub min_rarity: ItemRarity,
}

/// A weighted random drop table.
///
/// Given a set of weighted entries, `roll()` selects drops using weighted
/// random sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LootTable {
    /// Name / identifier of this loot table.
    pub name: String,
    /// Entries in the table.
    pub entries: Vec<LootEntry>,
    /// Number of rolls (how many items to pick).
    pub rolls: u32,
    /// Chance per roll that anything drops at all (0..1).
    pub drop_chance: f32,
    /// Global rarity modifier (multiplied with base weights).
    pub rarity_modifier: f32,
}

/// Result of rolling a loot table.
#[derive(Debug, Clone)]
pub struct LootDrop {
    /// Item id.
    pub item_id: String,
    /// Quantity.
    pub quantity: u32,
}

impl LootTable {
    /// Create a new loot table.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entries: Vec::new(),
            rolls: 1,
            drop_chance: 1.0,
            rarity_modifier: 1.0,
        }
    }

    /// Add an entry.
    pub fn add_entry(&mut self, entry: LootEntry) {
        self.entries.push(entry);
    }

    /// Add a simple entry with default rarity filter.
    pub fn add_simple(
        &mut self,
        item_id: impl Into<String>,
        weight: f32,
        min_qty: u32,
        max_qty: u32,
    ) {
        self.entries.push(LootEntry {
            item_id: item_id.into(),
            weight,
            min_quantity: min_qty,
            max_quantity: max_qty,
            min_rarity: ItemRarity::Common,
        });
    }

    /// Roll the loot table and return the drops.
    pub fn roll(&self, rng: &mut genovo_core::Rng) -> Vec<LootDrop> {
        let mut drops = Vec::new();

        for _ in 0..self.rolls {
            // Check drop chance.
            if rng.next_f32() > self.drop_chance {
                continue;
            }

            if self.entries.is_empty() {
                continue;
            }

            // Build weight array.
            let weights: Vec<f32> = self
                .entries
                .iter()
                .map(|e| e.weight * self.rarity_modifier)
                .collect();

            let index = rng.weighted_pick(&weights);
            let entry = &self.entries[index];

            // Random quantity.
            let quantity = if entry.min_quantity == entry.max_quantity {
                entry.min_quantity
            } else {
                rng.range_i32(entry.min_quantity as i32, entry.max_quantity as i32 + 1) as u32
            };

            // Try to merge with existing drops.
            if let Some(existing) = drops.iter_mut().find(|d: &&mut LootDrop| d.item_id == entry.item_id) {
                existing.quantity += quantity;
            } else {
                drops.push(LootDrop {
                    item_id: entry.item_id.clone(),
                    quantity,
                });
            }
        }

        drops
    }
}

// ---------------------------------------------------------------------------
// Crafting recipe
// ---------------------------------------------------------------------------

/// An ingredient required for a crafting recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CraftingIngredient {
    /// Item id.
    pub item_id: String,
    /// Quantity required.
    pub quantity: u32,
}

/// Output of a crafting recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CraftingOutput {
    /// Item id produced.
    pub item_id: String,
    /// Quantity produced.
    pub quantity: u32,
}

/// A crafting recipe that transforms input items into output items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CraftingRecipe {
    /// Unique recipe id.
    pub recipe_id: String,
    /// Display name.
    pub name: String,
    /// Required ingredients.
    pub ingredients: Vec<CraftingIngredient>,
    /// Output items.
    pub outputs: Vec<CraftingOutput>,
    /// Crafting time in seconds (0 = instant).
    pub craft_time: f32,
    /// Required crafting station type (if any).
    pub station_type: Option<String>,
    /// Required character level (0 = no requirement).
    pub required_level: u32,
    /// Whether this recipe has been unlocked/discovered.
    pub unlocked: bool,
}

impl CraftingRecipe {
    /// Create a new recipe.
    pub fn new(recipe_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            recipe_id: recipe_id.into(),
            name: name.into(),
            ingredients: Vec::new(),
            outputs: Vec::new(),
            craft_time: 0.0,
            station_type: None,
            required_level: 0,
            unlocked: true,
        }
    }

    /// Add an ingredient.
    pub fn add_ingredient(&mut self, item_id: impl Into<String>, quantity: u32) {
        self.ingredients.push(CraftingIngredient {
            item_id: item_id.into(),
            quantity,
        });
    }

    /// Add an output.
    pub fn add_output(&mut self, item_id: impl Into<String>, quantity: u32) {
        self.outputs.push(CraftingOutput {
            item_id: item_id.into(),
            quantity,
        });
    }

    /// Check whether the inventory has all required ingredients.
    pub fn can_craft(&self, inventory: &Inventory) -> bool {
        if !self.unlocked {
            return false;
        }
        self.ingredients
            .iter()
            .all(|ing| inventory.has_item(&ing.item_id, ing.quantity))
    }

    /// How many times this recipe can be crafted with the current inventory.
    pub fn max_craftable(&self, inventory: &Inventory) -> u32 {
        if !self.unlocked || self.ingredients.is_empty() {
            return 0;
        }
        self.ingredients
            .iter()
            .map(|ing| {
                let available = inventory.count_item(&ing.item_id);
                available / ing.quantity.max(1)
            })
            .min()
            .unwrap_or(0)
    }

    /// Execute the recipe: consume ingredients and add outputs.
    pub fn craft(
        &self,
        inventory: &mut Inventory,
        registry: &ItemRegistry,
    ) -> InventoryResult<Vec<ItemStack>> {
        if !self.can_craft(inventory) {
            return Err(InventoryError::Full); // Reusing error; could add a specific variant.
        }

        // Check if there's room for outputs.
        let output_slots_needed: usize = self
            .outputs
            .iter()
            .map(|o| {
                let max_stack = registry
                    .get(&o.item_id)
                    .map(|d| d.max_stack)
                    .unwrap_or(1);
                ((o.quantity as f32) / (max_stack as f32)).ceil() as usize
            })
            .sum();

        // Pessimistic slot check (ingredients may free slots).
        let ingredient_slots = self
            .ingredients
            .iter()
            .filter(|ing| {
                // Count how many slots will be fully consumed.
                let total = inventory.count_item(&ing.item_id);
                total <= ing.quantity
            })
            .count();

        let available_slots = inventory.free_slots() + ingredient_slots;
        if available_slots < output_slots_needed {
            return Err(InventoryError::Full);
        }

        // Consume ingredients.
        for ing in &self.ingredients {
            inventory.remove_item(&ing.item_id, ing.quantity);
        }

        // Add outputs.
        let mut produced = Vec::new();
        for output in &self.outputs {
            let leftover = inventory.add_item(&output.item_id, output.quantity, registry)?;
            produced.push(ItemStack::new(&output.item_id, output.quantity - leftover));
        }

        Ok(produced)
    }
}

/// Collection of crafting recipes.
#[derive(Debug, Default)]
pub struct CraftingBook {
    recipes: Vec<CraftingRecipe>,
}

impl CraftingBook {
    /// Create an empty crafting book.
    pub fn new() -> Self {
        Self {
            recipes: Vec::new(),
        }
    }

    /// Add a recipe.
    pub fn add_recipe(&mut self, recipe: CraftingRecipe) {
        self.recipes.push(recipe);
    }

    /// Get a recipe by id.
    pub fn get_recipe(&self, recipe_id: &str) -> Option<&CraftingRecipe> {
        self.recipes.iter().find(|r| r.recipe_id == recipe_id)
    }

    /// Get all craftable recipes given the current inventory.
    pub fn available_recipes(&self, inventory: &Inventory) -> Vec<&CraftingRecipe> {
        self.recipes
            .iter()
            .filter(|r| r.can_craft(inventory))
            .collect()
    }

    /// Number of recipes.
    pub fn len(&self) -> usize {
        self.recipes.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.recipes.is_empty()
    }

    /// All recipes.
    pub fn iter(&self) -> impl Iterator<Item = &CraftingRecipe> {
        self.recipes.iter()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_registry() -> ItemRegistry {
        let mut reg = ItemRegistry::new();
        reg.register(ItemDefinition {
            item_id: "wood".into(),
            display_name: "Wood".into(),
            category: ItemCategory::Material,
            max_stack: 64,
            weight: 0.5,
            base_value: 1,
            ..Default::default()
        });
        reg.register(ItemDefinition {
            item_id: "stone".into(),
            display_name: "Stone".into(),
            category: ItemCategory::Material,
            max_stack: 64,
            weight: 1.0,
            base_value: 2,
            ..Default::default()
        });
        reg.register(ItemDefinition {
            item_id: "iron_sword".into(),
            display_name: "Iron Sword".into(),
            category: ItemCategory::Weapon,
            rarity: ItemRarity::Uncommon,
            max_stack: 1,
            weight: 3.0,
            base_value: 100,
            equip_slot: Some(EquipmentSlot::MainHand),
            ..Default::default()
        });
        reg.register(ItemDefinition {
            item_id: "health_potion".into(),
            display_name: "Health Potion".into(),
            category: ItemCategory::Consumable,
            max_stack: 20,
            weight: 0.3,
            base_value: 10,
            consumable: true,
            ..Default::default()
        });
        reg.register(ItemDefinition {
            item_id: "iron_helmet".into(),
            display_name: "Iron Helmet".into(),
            category: ItemCategory::Armor,
            max_stack: 1,
            weight: 2.0,
            base_value: 50,
            equip_slot: Some(EquipmentSlot::Head),
            ..Default::default()
        });
        reg
    }

    #[test]
    fn add_and_count_items() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);

        let leftover = inv.add_item("wood", 30, &reg).unwrap();
        assert_eq!(leftover, 0);
        assert_eq!(inv.count_item("wood"), 30);
        assert_eq!(inv.used_slots(), 1);
    }

    #[test]
    fn add_items_stacks_correctly() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);

        inv.add_item("wood", 50, &reg).unwrap();
        inv.add_item("wood", 30, &reg).unwrap();

        // 50 + 30 = 80, max_stack = 64, so should be 64 + 16 = 2 slots.
        assert_eq!(inv.count_item("wood"), 80);
        assert_eq!(inv.used_slots(), 2);
    }

    #[test]
    fn remove_items() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 50, &reg).unwrap();

        let removed = inv.remove_item("wood", 20);
        assert_eq!(removed, 20);
        assert_eq!(inv.count_item("wood"), 30);
    }

    #[test]
    fn remove_more_than_available() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 10, &reg).unwrap();

        let removed = inv.remove_item("wood", 20);
        assert_eq!(removed, 10);
        assert_eq!(inv.count_item("wood"), 0);
        assert_eq!(inv.used_slots(), 0);
    }

    #[test]
    fn swap_slots() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 10, &reg).unwrap();
        inv.add_item("stone", 5, &reg).unwrap();

        inv.swap_slots(0, 1).unwrap();

        let slot0 = inv.get_slot(0).unwrap().unwrap();
        assert_eq!(slot0.item_id, "stone");
        let slot1 = inv.get_slot(1).unwrap().unwrap();
        assert_eq!(slot1.item_id, "wood");
    }

    #[test]
    fn split_stack() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 30, &reg).unwrap();

        inv.split_stack(0, 1, 10).unwrap();

        assert_eq!(inv.get_slot(0).unwrap().unwrap().quantity, 20);
        assert_eq!(inv.get_slot(1).unwrap().unwrap().quantity, 10);
    }

    #[test]
    fn merge_slots_combines() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 30, &reg).unwrap();
        inv.split_stack(0, 1, 10).unwrap();

        // Slot 0 has 20, slot 1 has 10.
        inv.merge_slots(1, 0, &reg).unwrap();
        assert_eq!(inv.get_slot(0).unwrap().unwrap().quantity, 30);
        assert!(inv.get_slot(1).unwrap().is_none());
    }

    #[test]
    fn weight_limit_enforced() {
        let reg = test_registry();
        let mut inv = Inventory::with_weight_limit(10, 10.0);

        // wood weighs 0.5 per unit, so 20 = 10.0 weight (at limit).
        let result = inv.add_item("wood", 20, &reg);
        assert!(result.is_ok());

        // Adding more should fail.
        let result = inv.add_item("wood", 1, &reg);
        assert!(matches!(result, Err(InventoryError::WeightLimitExceeded { .. })));
    }

    #[test]
    fn has_item() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 10, &reg).unwrap();

        assert!(inv.has_item("wood", 5));
        assert!(inv.has_item("wood", 10));
        assert!(!inv.has_item("wood", 11));
        assert!(!inv.has_item("stone", 1));
    }

    #[test]
    fn find_item_and_empty_slot() {
        let reg = test_registry();
        let mut inv = Inventory::new(5);
        inv.add_item("wood", 10, &reg).unwrap();

        assert_eq!(inv.find_item("wood"), Some(0));
        assert_eq!(inv.find_item("stone"), None);
        assert_eq!(inv.find_empty_slot(), Some(1));
    }

    #[test]
    fn equipment_equip_unequip() {
        let reg = test_registry();
        let mut equip = Equipment::new();

        let sword = ItemStack::new("iron_sword", 1);
        let prev = equip.equip(sword, EquipmentSlot::MainHand, &reg).unwrap();
        assert!(prev.is_none());
        assert!(equip.is_equipped(EquipmentSlot::MainHand));

        let removed = equip.unequip(EquipmentSlot::MainHand);
        assert!(removed.is_some());
        assert!(!equip.is_equipped(EquipmentSlot::MainHand));
    }

    #[test]
    fn equipment_wrong_slot() {
        let reg = test_registry();
        let mut equip = Equipment::new();

        let helmet = ItemStack::new("iron_helmet", 1);
        let result = equip.equip(helmet, EquipmentSlot::MainHand, &reg);
        assert!(matches!(result, Err(InventoryError::WrongEquipSlot(_, _))));
    }

    #[test]
    fn loot_table_roll() {
        let mut table = LootTable::new("test_loot");
        table.add_simple("wood", 10.0, 1, 5);
        table.add_simple("stone", 5.0, 1, 3);
        table.rolls = 3;

        let mut rng = genovo_core::Rng::new(42);
        let drops = table.roll(&mut rng);
        assert!(!drops.is_empty(), "Should produce at least one drop");

        for drop in &drops {
            assert!(
                drop.item_id == "wood" || drop.item_id == "stone",
                "Unexpected item: {}",
                drop.item_id
            );
            assert!(drop.quantity >= 1);
        }
    }

    #[test]
    fn loot_table_deterministic() {
        let mut table = LootTable::new("test_loot");
        table.add_simple("wood", 10.0, 1, 5);
        table.add_simple("stone", 5.0, 1, 3);
        table.rolls = 5;

        let mut rng1 = genovo_core::Rng::new(42);
        let drops1 = table.roll(&mut rng1);

        let mut rng2 = genovo_core::Rng::new(42);
        let drops2 = table.roll(&mut rng2);

        assert_eq!(drops1.len(), drops2.len());
        for (a, b) in drops1.iter().zip(drops2.iter()) {
            assert_eq!(a.item_id, b.item_id);
            assert_eq!(a.quantity, b.quantity);
        }
    }

    #[test]
    fn crafting_recipe_can_craft() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 5, &reg).unwrap();
        inv.add_item("stone", 3, &reg).unwrap();

        let mut recipe = CraftingRecipe::new("wooden_pickaxe", "Wooden Pickaxe");
        recipe.add_ingredient("wood", 3);
        recipe.add_ingredient("stone", 2);
        recipe.add_output("iron_sword", 1); // Placeholder output

        assert!(recipe.can_craft(&inv));
        assert_eq!(recipe.max_craftable(&inv), 1);
    }

    #[test]
    fn crafting_recipe_cannot_craft_missing_ingredients() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 1, &reg).unwrap();

        let mut recipe = CraftingRecipe::new("test", "Test");
        recipe.add_ingredient("wood", 5);
        recipe.add_output("stone", 1);

        assert!(!recipe.can_craft(&inv));
    }

    #[test]
    fn crafting_recipe_execute() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 10, &reg).unwrap();
        inv.add_item("stone", 5, &reg).unwrap();

        let mut recipe = CraftingRecipe::new("test", "Test Recipe");
        recipe.add_ingredient("wood", 3);
        recipe.add_ingredient("stone", 2);
        recipe.add_output("health_potion", 2);

        let result = recipe.craft(&mut inv, &reg);
        assert!(result.is_ok());

        assert_eq!(inv.count_item("wood"), 7);
        assert_eq!(inv.count_item("stone"), 3);
        assert_eq!(inv.count_item("health_potion"), 2);
    }

    #[test]
    fn inventory_sort() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("health_potion", 5, &reg).unwrap();
        inv.add_item("wood", 10, &reg).unwrap();
        inv.add_item("iron_sword", 1, &reg).unwrap();
        inv.add_item("stone", 3, &reg).unwrap();

        inv.sort(&reg);

        // Weapons should come first, then consumables, then materials.
        let items: Vec<String> = inv.iter().map(|(_, s)| s.item_id.clone()).collect();
        assert_eq!(items[0], "iron_sword");
    }

    #[test]
    fn item_stack_split() {
        let mut stack = ItemStack::new("wood", 20);
        let split = stack.split(5);
        assert!(split.is_some());
        let split = split.unwrap();
        assert_eq!(split.quantity, 5);
        assert_eq!(stack.quantity, 15);
    }

    #[test]
    fn item_stack_split_all_returns_none() {
        let mut stack = ItemStack::new("wood", 10);
        let split = stack.split(10);
        assert!(split.is_none());
        assert_eq!(stack.quantity, 10); // Unchanged.
    }

    #[test]
    fn crafting_book() {
        let reg = test_registry();
        let mut inv = Inventory::new(10);
        inv.add_item("wood", 20, &reg).unwrap();

        let mut book = CraftingBook::new();
        let mut r1 = CraftingRecipe::new("r1", "Recipe 1");
        r1.add_ingredient("wood", 5);
        r1.add_output("stone", 1);
        book.add_recipe(r1);

        let mut r2 = CraftingRecipe::new("r2", "Recipe 2");
        r2.add_ingredient("iron_sword", 1);
        r2.add_output("wood", 5);
        book.add_recipe(r2);

        let available = book.available_recipes(&inv);
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].recipe_id, "r1");
    }

    #[test]
    fn rarity_ordering() {
        assert!(ItemRarity::Common < ItemRarity::Uncommon);
        assert!(ItemRarity::Rare < ItemRarity::Epic);
        assert!(ItemRarity::Legendary < ItemRarity::Mythic);
    }

    #[test]
    fn rarity_base_weight() {
        assert!(ItemRarity::Common.base_weight() > ItemRarity::Legendary.base_weight());
    }

    #[test]
    fn item_total_weight() {
        let reg = test_registry();
        let stack = ItemStack::new("wood", 10);
        assert!((stack.total_weight(&reg) - 5.0).abs() < 0.01);
    }

    #[test]
    fn item_total_value() {
        let reg = test_registry();
        let stack = ItemStack::new("stone", 5);
        assert_eq!(stack.total_value(&reg), 10);
    }
}
